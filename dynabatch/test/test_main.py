"""Tests for dynabatch/main.py internals and public components."""

import numpy as np
import pytest

from dynabatch.main import (
    MaxTokenBatchSampler,
    TextDataset,
    _collate_fn,
    _select_optimal_batch_size,
    _tokenize_chunk,
    compute_lengths,
)

# ---------------------------------------------------------------------------
# TextDataset
# ---------------------------------------------------------------------------


def test_text_dataset_len(sample_texts):
    ds = TextDataset(sample_texts)
    assert len(ds) == len(sample_texts)


def test_text_dataset_getitem(sample_texts):
    ds = TextDataset(sample_texts)
    for i, text in enumerate(sample_texts):
        item = ds[i]
        assert isinstance(item, dict)
        assert item["text"] == text


# ---------------------------------------------------------------------------
# _collate_fn
# ---------------------------------------------------------------------------

_COLLATE_MAX_LENGTH = 128


def test_collate_fn_keys(sample_texts, mock_tokenizer):
    batch = [{"text": t} for t in sample_texts[:3]]
    result = _collate_fn(batch, tokenizer=mock_tokenizer, max_length=_COLLATE_MAX_LENGTH)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "texts" in result
    assert isinstance(result["texts"], list)


def test_collate_fn_texts_preserved(sample_texts, mock_tokenizer):
    subset = sample_texts[:3]
    batch = [{"text": t} for t in subset]
    result = _collate_fn(batch, tokenizer=mock_tokenizer, max_length=_COLLATE_MAX_LENGTH)
    assert result["texts"] == subset


def test_collate_fn_tensor_shapes(sample_texts, mock_tokenizer):
    import torch

    batch = [{"text": t} for t in sample_texts[:3]]
    result = _collate_fn(batch, tokenizer=mock_tokenizer, max_length=_COLLATE_MAX_LENGTH)
    assert result["input_ids"].shape[0] == 3
    assert result["attention_mask"].shape[0] == 3
    assert result["input_ids"].dtype == torch.long


# ---------------------------------------------------------------------------
# _tokenize_chunk / compute_lengths
# ---------------------------------------------------------------------------


def test_tokenize_chunk_returns_lengths(sample_texts, mock_tokenizer):
    chunk = sample_texts[:3]
    lengths: list[tuple[int, int, int]] = []
    for t in chunk:
        lengths.extend(_tokenize_chunk(t, tokenizer=mock_tokenizer, max_length=128))
    assert len(lengths) == 3
    for token_len, word_len, char_len in lengths:
        assert isinstance(token_len, int) and token_len > 0
        assert isinstance(word_len, int) and word_len > 0
        assert isinstance(char_len, int) and char_len > 0


def test_tokenize_chunk_truncation(mock_tokenizer):
    text = "word " * 20  # 20 tokens
    rows = _tokenize_chunk(text, tokenizer=mock_tokenizer, max_length=5)
    token_len, _word_len, _char_len = rows[0]
    assert token_len <= 5


def test_compute_sequence_lengths_count(sample_texts, mock_tokenizer):
    lengths = compute_lengths(sample_texts, mock_tokenizer, max_length=64)
    assert len(lengths) == len(sample_texts)


def test_compute_sequence_lengths_positive(sample_texts, mock_tokenizer):
    lengths = compute_lengths(sample_texts, mock_tokenizer, max_length=64)
    for token_len, word_len, char_len in lengths:
        assert token_len > 0 and word_len > 0 and char_len > 0


def test_compute_sequence_lengths_bounded(sample_texts, mock_tokenizer):
    max_len = 4
    lengths = compute_lengths(sample_texts, mock_tokenizer, max_length=max_len)
    for token_len, _word_len, _char_len in lengths:
        assert token_len <= max_len


# ---------------------------------------------------------------------------
# _select_optimal_batch_size
# ---------------------------------------------------------------------------

# Representative sorted descending rows: (token_len, word_len, char_len)
_TOKEN_SEQ = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
_SEQ_LENGTHS = np.array([[t, max(1, t // 2), t * 5] for t in _TOKEN_SEQ], dtype=np.int64)
_BASELINE_BS = 2
_BASELINE_MAX_TOKEN = np.int64(_SEQ_LENGTHS[0, 0])
_BASELINE_MAX_WORD = np.int64(_SEQ_LENGTHS[0, 1])
_BASELINE_MAX_CHAR = np.int64(_SEQ_LENGTHS[0, 2])
_BASELINE_TOTAL_TOKENS = np.int64(_SEQ_LENGTHS[:_BASELINE_BS, 0].sum())


def test_select_optimal_batch_size_returns_int():
    result = _select_optimal_batch_size(
        max_input_length=64,
        sequence_lengths=_SEQ_LENGTHS,
        baseline_max_token_len=_BASELINE_MAX_TOKEN,
        baseline_max_word_len=_BASELINE_MAX_WORD,
        baseline_max_char_len=_BASELINE_MAX_CHAR,
        baseline_batch_size=_BASELINE_BS,
        baseline_total_tokens=_BASELINE_TOTAL_TOKENS,
    )
    assert isinstance(result, int)


def test_select_optimal_batch_size_bounded():
    result = _select_optimal_batch_size(
        max_input_length=64,
        sequence_lengths=_SEQ_LENGTHS,
        baseline_max_token_len=_BASELINE_MAX_TOKEN,
        baseline_max_word_len=_BASELINE_MAX_WORD,
        baseline_max_char_len=_BASELINE_MAX_CHAR,
        baseline_batch_size=_BASELINE_BS,
        baseline_total_tokens=_BASELINE_TOTAL_TOKENS,
    )
    assert 1 <= result <= len(_SEQ_LENGTHS)


def test_select_optimal_batch_size_fallback_to_baseline():
    """With threshold=0.0 no candidate can satisfy the constraint; returns baseline."""
    result = _select_optimal_batch_size(
        max_input_length=64,
        sequence_lengths=_SEQ_LENGTHS,
        baseline_max_token_len=_BASELINE_MAX_TOKEN,
        baseline_max_word_len=_BASELINE_MAX_WORD,
        baseline_max_char_len=_BASELINE_MAX_CHAR,
        baseline_batch_size=_BASELINE_BS,
        baseline_total_tokens=_BASELINE_TOTAL_TOKENS,
        threshold=0.0,
    )
    assert result == min(_BASELINE_BS, len(_SEQ_LENGTHS))


def test_select_optimal_batch_size_permissive_threshold():
    """With a very high threshold, the result should be at least the baseline."""
    result = _select_optimal_batch_size(
        max_input_length=64,
        sequence_lengths=_SEQ_LENGTHS,
        baseline_max_token_len=_BASELINE_MAX_TOKEN,
        baseline_max_word_len=_BASELINE_MAX_WORD,
        baseline_max_char_len=_BASELINE_MAX_CHAR,
        baseline_batch_size=_BASELINE_BS,
        baseline_total_tokens=_BASELINE_TOTAL_TOKENS,
        threshold=1.0,
    )
    assert result >= _BASELINE_BS


# ---------------------------------------------------------------------------
# MaxTokenBatchSampler
# ---------------------------------------------------------------------------


def _make_sampler(sequence_lengths, min_batch_size=2, shuffle=False, seed=21) -> MaxTokenBatchSampler:
    return MaxTokenBatchSampler(
        sequence_lengths=sequence_lengths,
        max_input_length=64,
        min_batch_size=min_batch_size,
        shuffle=shuffle,
        shuffle_seed=seed,
    )


def test_sampler_covers_all_indices(sample_texts, mock_tokenizer):
    lengths = compute_lengths(sample_texts, mock_tokenizer, max_length=64)
    sampler = _make_sampler(lengths)
    all_indices = [idx for batch in sampler for idx in batch]
    assert sorted(all_indices) == list(range(len(sample_texts)))


def test_sampler_first_batch_size(sample_texts, mock_tokenizer):
    min_bs = 2
    lengths = compute_lengths(sample_texts, mock_tokenizer, max_length=64)
    sampler = _make_sampler(lengths, min_batch_size=min_bs)
    first_batch = list(sampler)[0]
    assert len(first_batch) == min_bs


def test_sampler_len(sample_texts, mock_tokenizer):
    lengths = compute_lengths(sample_texts, mock_tokenizer, max_length=64)
    sampler = _make_sampler(lengths)
    batches = list(sampler)
    assert len(sampler) == len(batches)


def test_sampler_no_overlap(sample_texts, mock_tokenizer):
    lengths = compute_lengths(sample_texts, mock_tokenizer, max_length=64)
    sampler = _make_sampler(lengths)
    all_indices = [idx for batch in sampler for idx in batch]
    assert len(all_indices) == len(set(all_indices)), "Indices should not repeat across batches"


def test_sampler_shuffle_deterministic(sample_texts, mock_tokenizer):
    lengths = compute_lengths(sample_texts, mock_tokenizer, max_length=64)
    sampler_a = _make_sampler(lengths, shuffle=True, seed=42)
    sampler_b = _make_sampler(lengths, shuffle=True, seed=42)
    assert list(sampler_a) == list(sampler_b)


def test_sampler_shuffle_vs_no_shuffle(sample_texts, mock_tokenizer):
    lengths = compute_lengths(sample_texts, mock_tokenizer, max_length=64)
    sampler_no_shuffle = _make_sampler(lengths, shuffle=False)

    # shuffle_keep_first_n=0 ensures ALL batches are eligible for reordering
    sampler_shuffle = MaxTokenBatchSampler(
        sequence_lengths=lengths,
        max_input_length=64,
        min_batch_size=2,
        shuffle=True,
        shuffle_seed=0,
        shuffle_keep_first_n=0,
    )

    unshuffled = list(sampler_no_shuffle)
    shuffled = list(sampler_shuffle)

    # All indices still covered after shuffle
    assert sorted(idx for batch in shuffled for idx in batch) == list(range(len(sample_texts)))

    # With keep_first_n=0, all batches past index 0 are shuffled; order should differ
    # (only assert when there are 3+ batches, otherwise shuffle may be identity)
    if len(unshuffled) >= 3:
        assert shuffled != unshuffled, "Shuffle should change batch order with keep_first_n=0"
