"""Tests for dynabatch internals and public components."""

import numpy as np
import pytest

from dynabatch.main import TextDataset, _collate_fn, compute_lengths
from dynabatch.regressor import build_baseline_features, select_optimal_batch_size
from dynabatch.sampler import MaxTokenBatchSampler

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


def test_collate_fn_keys(sample_texts, mock_tokenizer):
    batch = [{"text": t} for t in sample_texts[:3]]
    result = _collate_fn(batch, tokenizer=mock_tokenizer)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "texts" in result
    assert isinstance(result["texts"], list)


def test_collate_fn_texts_preserved(sample_texts, mock_tokenizer):
    subset = sample_texts[:3]
    batch = [{"text": t} for t in subset]
    result = _collate_fn(batch, tokenizer=mock_tokenizer)
    assert result["texts"] == subset


def test_collate_fn_tensor_shapes(sample_texts, mock_tokenizer):
    import torch

    batch = [{"text": t} for t in sample_texts[:3]]
    result = _collate_fn(batch, tokenizer=mock_tokenizer)
    assert result["input_ids"].shape[0] == 3
    assert result["attention_mask"].shape[0] == 3
    assert result["input_ids"].dtype == torch.long


# ---------------------------------------------------------------------------
# compute_lengths
# ---------------------------------------------------------------------------


def test_compute_lengths_return_counts(sample_texts, precomputed_lengths):
    token_lengths, word_lengths, char_lengths, truncated_texts = precomputed_lengths
    assert len(token_lengths) == len(sample_texts)
    assert len(word_lengths) == len(sample_texts)
    assert len(char_lengths) == len(sample_texts)
    assert len(truncated_texts) == len(sample_texts)


def test_compute_lengths_positive_values(precomputed_lengths):
    token_lengths, word_lengths, char_lengths, _ = precomputed_lengths
    for tl, wl, cl in zip(token_lengths, word_lengths, char_lengths):
        assert tl > 0 and wl > 0 and cl > 0


def test_compute_lengths_bounded_by_max_length(sample_texts, mock_tokenizer):
    token_lengths, _, _, _ = compute_lengths(sample_texts[:10], mock_tokenizer, max_length=4)
    for tl in token_lengths:
        assert tl <= 4


def test_compute_lengths_truncated_texts_are_strings(precomputed_lengths):
    _, _, _, truncated_texts = precomputed_lengths
    for t in truncated_texts:
        assert isinstance(t, str)


# ---------------------------------------------------------------------------
# select_optimal_batch_size
# ---------------------------------------------------------------------------

# Representative sorted descending values (longest first), as in the sampler.
_TOKEN_SEQ = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
_TOKEN_LENGTHS = np.array(_TOKEN_SEQ, dtype=np.int64)
_WORD_LENGTHS = np.array([max(1, t // 2) for t in _TOKEN_SEQ], dtype=np.int64)
_CHAR_LENGTHS = np.array([t * 5 for t in _TOKEN_SEQ], dtype=np.int64)
_BASELINE_BS = 2
_CANDIDATE_BATCH_SIZES = np.array([2, 3, 4, 5, 6])
# Remaining sequences after the first (baseline) batch — matches sampler calls.
_TOKEN_REM = _TOKEN_LENGTHS[_BASELINE_BS:]
_WORD_REM = _WORD_LENGTHS[_BASELINE_BS:]
_CHAR_REM = _CHAR_LENGTHS[_BASELINE_BS:]


def _call_select_optimal(
    *,
    threshold: float = 0.75,
    token_lengths=None,
    word_lengths=None,
    char_lengths=None,
    candidate_batch_sizes=None,
    baseline_bs=None,
) -> int:
    tl = _TOKEN_REM if token_lengths is None else token_lengths
    wl = _WORD_REM if word_lengths is None else word_lengths
    cl = _CHAR_REM if char_lengths is None else char_lengths
    cands = _CANDIDATE_BATCH_SIZES if candidate_batch_sizes is None else candidate_batch_sizes
    bs = _BASELINE_BS if baseline_bs is None else baseline_bs
    baseline = build_baseline_features(_TOKEN_LENGTHS, _WORD_LENGTHS, _CHAR_LENGTHS, bs, len(cands))
    return select_optimal_batch_size(
        token_lengths=tl,
        word_lengths=wl,
        char_lengths=cl,
        baseline_features=baseline,
        threshold=threshold,
        candidate_batch_sizes=cands,
    )


def test_select_optimal_batch_size_returns_int():
    result = _call_select_optimal()
    assert isinstance(result, int)


def test_select_optimal_batch_size_bounded():
    result = _call_select_optimal()
    assert 1 <= result <= int(max(_CANDIDATE_BATCH_SIZES))


def test_select_optimal_batch_size_fallback_to_baseline():
    """With threshold=-1.0 no candidate passes; returns baseline batch size."""
    result = _call_select_optimal(threshold=-1.0)
    assert result == _BASELINE_BS


def test_select_optimal_batch_size_permissive_threshold():
    """Stub regressor predicts 0; all candidates pass — largest is chosen."""
    result = _call_select_optimal(threshold=1.0)
    assert result == int(max(_CANDIDATE_BATCH_SIZES))
    assert result >= _BASELINE_BS


# ---------------------------------------------------------------------------
# MaxTokenBatchSampler
# ---------------------------------------------------------------------------


def _make_sampler(
    token_lengths, word_lengths, char_lengths, min_batch_size=2, shuffle=False, seed=21
) -> MaxTokenBatchSampler:
    return MaxTokenBatchSampler(
        token_lengths=token_lengths,
        word_lengths=word_lengths,
        char_lengths=char_lengths,
        min_batch_size=min_batch_size,
        shuffle=shuffle,
        shuffle_seed=seed,
    )


def test_sampler_covers_all_indices(sample_texts, precomputed_lengths):
    token_lengths, word_lengths, char_lengths, _ = precomputed_lengths
    sampler = _make_sampler(token_lengths, word_lengths, char_lengths)
    all_indices = [idx for batch in sampler for idx in batch]
    assert sorted(all_indices) == list(range(len(sample_texts)))


def test_sampler_first_batch_size(precomputed_lengths):
    min_bs = 2
    token_lengths, word_lengths, char_lengths, _ = precomputed_lengths
    sampler = _make_sampler(token_lengths, word_lengths, char_lengths, min_batch_size=min_bs)
    first_batch = list(sampler)[0]
    assert len(first_batch) == min_bs


def test_sampler_len(precomputed_lengths):
    token_lengths, word_lengths, char_lengths, _ = precomputed_lengths
    sampler = _make_sampler(token_lengths, word_lengths, char_lengths)
    batches = list(sampler)
    assert len(sampler) == len(batches)


def test_sampler_no_overlap(precomputed_lengths):
    token_lengths, word_lengths, char_lengths, _ = precomputed_lengths
    sampler = _make_sampler(token_lengths, word_lengths, char_lengths)
    all_indices = [idx for batch in sampler for idx in batch]
    assert len(all_indices) == len(set(all_indices)), "Indices should not repeat across batches"


def test_sampler_shuffle_deterministic(precomputed_lengths):
    token_lengths, word_lengths, char_lengths, _ = precomputed_lengths
    sampler_a = _make_sampler(token_lengths, word_lengths, char_lengths, shuffle=True, seed=42)
    sampler_b = _make_sampler(token_lengths, word_lengths, char_lengths, shuffle=True, seed=42)
    assert list(sampler_a) == list(sampler_b)


def test_sampler_shuffle_vs_no_shuffle(sample_texts, precomputed_lengths):
    token_lengths, word_lengths, char_lengths, _ = precomputed_lengths
    sampler_no_shuffle = _make_sampler(token_lengths, word_lengths, char_lengths, shuffle=False)

    sampler_shuffle = MaxTokenBatchSampler(
        token_lengths=token_lengths,
        word_lengths=word_lengths,
        char_lengths=char_lengths,
        min_batch_size=2,
        shuffle=True,
        shuffle_seed=0,
        shuffle_keep_first_n=0,
    )

    unshuffled = list(sampler_no_shuffle)
    shuffled = list(sampler_shuffle)

    assert sorted(idx for batch in shuffled for idx in batch) == list(range(len(sample_texts)))

    if len(unshuffled) >= 3:
        assert shuffled != unshuffled, "Shuffle should change batch order with keep_first_n=0"


def test_batch_size_increased(sample_texts_5000, mock_word_tokenizer_session):
    from dynabatch.main import build_dynamic_batch_dataloader

    min_batch_size = 32
    loader = build_dynamic_batch_dataloader(
        texts=sample_texts_5000,
        tokenizer=mock_word_tokenizer_session,
        batch_size=min_batch_size,
        max_input_token_length=512,
        max_batch_range=2.0,
        threshold=0.9,
        shuffle=False,
        num_workers=0,
    )
    batch_sizes = []

    for batch in loader:
        batch_sizes.append(len(batch["input_ids"]))

    assert max(batch_sizes) > min_batch_size
