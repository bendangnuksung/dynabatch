"""Tests for dynabatch internals and public components."""

import random
import re

import numpy as np
import pytest

import dynabatch.main as _main_module
from dynabatch.main import TextDataset, _collate_fn, compute_lengths, dynabatch_sampler
from dynabatch.regressor import _build_candidate_features, build_baseline_features, select_optimal_batch_size
from dynabatch.sampler import DynaBatchSampler


class _SimplePiece:
    def __init__(self, end: int):
        self.end = end


class _SimpleImmutableProto:
    def __init__(self, ends: list[int]):
        self.pieces = [_SimplePiece(end) for end in ends]


class _SentencePieceLikeModel:
    def encode_as_immutable_proto(self, text: str):
        ends: list[int] = []
        for match in re.finditer(r"\S+", text):
            end_char = match.end()
            ends.append(len(text[:end_char].encode("utf-8")))
        return _SimpleImmutableProto(ends)


class _SentencePieceFallbackTokenizer:
    def __init__(self):
        self.sp_model = _SentencePieceLikeModel()

    def __call__(
        self,
        text: str | list[str] | None = None,
        padding=False,
        truncation=False,
        max_length=None,
        return_offsets_mapping=False,
        return_attention_mask=True,
        **kwargs,
    ):
        if text is None:
            raise TypeError("text is required")
        texts = [text] if isinstance(text, str) else list(text)

        tokenized = [t.split() for t in texts]
        max_text_tokens = None
        if truncation and max_length is not None:
            max_text_tokens = max(max_length - 2, 0)
            tokenized = [tokens[:max_text_tokens] for tokens in tokenized]

        if return_offsets_mapping:
            raise ValueError("offset mapping unavailable for this tokenizer")

        self._last_batch_words = [list(tokens) for tokens in tokenized]

        input_ids = [[101] + [i + 1000 for i, _ in enumerate(tokens)] + [102] for tokens in tokenized]
        attention_masks = [[1] * len(ids) for ids in input_ids]

        if padding:
            max_len = max((len(ids) for ids in input_ids), default=0)
            input_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
            attention_masks = [mask + [0] * (max_len - len(mask)) for mask in attention_masks]

        if isinstance(text, str):
            result = {"input_ids": input_ids[0]}
            if return_attention_mask:
                result["attention_mask"] = attention_masks[0]
            return result

        result = {"input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = attention_masks
        return result

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=True):
        return [1 if token_id in (101, 102) else 0 for token_id in token_ids]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ) -> str:
        return self.batch_decode(
            [token_ids],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )[0]

    def batch_decode(
        self,
        sequences: list[list[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ) -> list[str]:
        words_batch = getattr(self, "_last_batch_words", None)
        if words_batch is None or len(words_batch) != len(sequences):
            return ["" for _ in sequences]
        out: list[str] = []
        for words in words_batch:
            out.append(" ".join(words))
        return out


class _SingleOnlyOffsetTokenizer:
    def __call__(
        self,
        text: str | list[str] | None = None,
        padding=False,
        truncation=False,
        max_length=None,
        return_offsets_mapping=False,
        **kwargs,
    ):
        if text is None:
            raise TypeError("text is required")
        if isinstance(text, list):
            if return_offsets_mapping:
                raise ValueError("batch offsets unavailable")
            tokenized = [t.split() for t in text]
            if truncation and max_length is not None:
                tokenized = [tokens[:max_length] for tokens in tokenized]
            input_ids = [[i + 1 for i, _ in enumerate(tokens)] for tokens in tokenized]
            attention_masks = [[1] * len(ids) for ids in input_ids]
            if padding:
                max_len = max((len(ids) for ids in input_ids), default=0)
                input_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
                attention_masks = [mask + [0] * (max_len - len(mask)) for mask in attention_masks]
            return {"input_ids": input_ids, "attention_mask": attention_masks}

        words = text.split()
        if truncation and max_length is not None:
            words = words[:max_length]

        input_ids = [i + 1 for i, _ in enumerate(words)]
        attention_mask = [1] * len(input_ids)
        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_offsets_mapping:
            offsets = []
            pos = 0
            for word in words:
                idx = text.find(word, pos)
                offsets.append((idx, idx + len(word)))
                pos = idx + len(word)
            result["offset_mapping"] = offsets
        return result


class _NoExactOffsetTokenizer:
    def __call__(
        self,
        text: str | list[str] | None = None,
        return_offsets_mapping=False,
        **kwargs,
    ):
        if return_offsets_mapping:
            raise ValueError("offset mapping unsupported")
        if text is None:
            raise TypeError("text is required")
        if isinstance(text, str):
            words = text.split()
            return {"input_ids": [i + 1 for i, _ in enumerate(words)], "attention_mask": [1] * len(words)}
        tokenized = [t.split() for t in text]
        return {
            "input_ids": [[i + 1 for i, _ in enumerate(tokens)] for tokens in tokenized],
            "attention_mask": [[1] * len(tokens) for tokens in tokenized],
        }

    def batch_decode(self, *args, **kwargs):
        raise RuntimeError(
            "Unable to compute exact char_lengths/word_lengths without offset mapping or a reliable decode."
        )

    def decode(self, token_ids, **kwargs):
        return self.batch_decode([token_ids], **kwargs)


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


def test_compute_lengths_fallback_sentencepiece_offsets():
    tokenizer = _SentencePieceFallbackTokenizer()
    texts = ["alpha beta gamma", "delta epsilon zeta", "你好 世界 再见"]

    token_lengths, word_lengths, char_lengths, truncated_texts = compute_lengths(
        texts,
        tokenizer=tokenizer,
        max_length=4,
        max_workers=1,
    )

    assert token_lengths == [4, 4, 4]
    assert truncated_texts[0].strip() == "alpha beta"
    assert truncated_texts[1].strip() == "delta epsilon"
    assert truncated_texts[2].strip() == "你好 世界"
    assert word_lengths == [2, 2, 2]
    assert char_lengths == [len("alpha beta") + 1, len("delta epsilon") + 1, len("你好 世界") + 1]  # +1 for the space


def test_compute_lengths_raises_when_no_exact_offset_strategy():
    tokenizer = _NoExactOffsetTokenizer()
    with pytest.raises(RuntimeError, match="Unable to compute exact char_lengths/word_lengths"):
        compute_lengths(
            ["a b c"],
            tokenizer=tokenizer,
            max_length=2,
            max_workers=1,
        )


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


def test_build_candidate_features_vectorized_parity():
    candidate_sizes = np.array([2, 3, 4, 5, 6], dtype=np.int64)
    token_lengths = np.array([20, 18, 16, 14, 12, 10, 8], dtype=np.int64)
    word_lengths = np.array([10, 9, 8, 7, 6, 5, 4], dtype=np.int64)
    char_lengths = np.array([100, 90, 80, 70, 60, 50, 40], dtype=np.int64)

    actual = _build_candidate_features(
        token_lengths=token_lengths,
        word_lengths=word_lengths,
        char_lengths=char_lengths,
        candidate_batch_sizes=candidate_sizes,
    )

    expected = {
        "batch_size_y": list(candidate_sizes),
        "token_mean_y": [float(np.mean(token_lengths[:bs])) for bs in candidate_sizes],
        "token_std_y": [float(np.std(token_lengths[:bs])) for bs in candidate_sizes],
        "token_sum_y": [float(np.sum(token_lengths[:bs])) for bs in candidate_sizes],
        "token_max_y": [float(np.max(token_lengths[:bs])) for bs in candidate_sizes],
        "word_mean_y": [float(np.mean(word_lengths[:bs])) for bs in candidate_sizes],
        "word_sum_y": [float(np.sum(word_lengths[:bs])) for bs in candidate_sizes],
        "word_max_y": [float(np.max(word_lengths[:bs])) for bs in candidate_sizes],
        "char_sum_y": [float(np.sum(char_lengths[:bs])) for bs in candidate_sizes],
    }

    assert expected.keys() == actual.keys()
    for key, expected_values in expected.items():
        np.testing.assert_allclose(actual[key], expected_values, rtol=1e-9, atol=1e-9)


def test_build_candidate_features_vectorized_parity_with_oversize_candidates():
    candidate_sizes = np.array([2, 3, 8, 12], dtype=np.int64)
    token_lengths = np.array([20, 18, 16, 14, 12], dtype=np.int64)
    word_lengths = np.array([10, 9, 8, 7, 6], dtype=np.int64)
    char_lengths = np.array([100, 90, 80, 70, 60], dtype=np.int64)

    actual = _build_candidate_features(
        token_lengths=token_lengths,
        word_lengths=word_lengths,
        char_lengths=char_lengths,
        candidate_batch_sizes=candidate_sizes,
    )

    expected = {
        "batch_size_y": list(candidate_sizes),
        "token_mean_y": [float(np.mean(token_lengths[:bs])) for bs in candidate_sizes],
        "token_std_y": [float(np.std(token_lengths[:bs])) for bs in candidate_sizes],
        "token_sum_y": [float(np.sum(token_lengths[:bs])) for bs in candidate_sizes],
        "token_max_y": [float(np.max(token_lengths[:bs])) for bs in candidate_sizes],
        "word_mean_y": [float(np.mean(word_lengths[:bs])) for bs in candidate_sizes],
        "word_sum_y": [float(np.sum(word_lengths[:bs])) for bs in candidate_sizes],
        "word_max_y": [float(np.max(word_lengths[:bs])) for bs in candidate_sizes],
        "char_sum_y": [float(np.sum(char_lengths[:bs])) for bs in candidate_sizes],
    }

    assert expected.keys() == actual.keys()
    for key, expected_values in expected.items():
        np.testing.assert_allclose(actual[key], expected_values, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# MaxTokenBatchSampler
# ---------------------------------------------------------------------------


def _make_sampler(
    token_lengths, word_lengths, char_lengths, min_batch_size=2, shuffle=False, seed=21
) -> DynaBatchSampler:
    return DynaBatchSampler(
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


def test_sampler_shuffle_vs_no_shuffle(sample_texts, precomputed_lengths):
    token_lengths, word_lengths, char_lengths, _ = precomputed_lengths
    sampler_no_shuffle = _make_sampler(token_lengths, word_lengths, char_lengths, shuffle=False)

    sampler_shuffle = DynaBatchSampler(
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


def test_sampler_does_not_mutate_global_random_state(precomputed_lengths):
    token_lengths, word_lengths, char_lengths, _ = precomputed_lengths
    random.seed(2026)
    expected_after = random.random()
    random.seed(2026)

    _ = DynaBatchSampler(
        token_lengths=token_lengths,
        word_lengths=word_lengths,
        char_lengths=char_lengths,
        min_batch_size=2,
        shuffle=True,
        shuffle_seed=999,
    )
    actual_after = random.random()
    assert actual_after == expected_after


def test_sampler_max_batch_range_one_does_not_crash(precomputed_lengths):
    token_lengths, word_lengths, char_lengths, _ = precomputed_lengths
    dynamic_sampler = DynaBatchSampler(
        token_lengths=token_lengths,
        word_lengths=word_lengths,
        char_lengths=char_lengths,
        min_batch_size=2,
        max_batch_range=1.0,
        dynamic_batch_mode=True,
    )
    static_sampler = DynaBatchSampler(
        token_lengths=token_lengths,
        word_lengths=word_lengths,
        char_lengths=char_lengths,
        min_batch_size=2,
        dynamic_batch_mode=False,
    )
    assert [len(batch) for batch in dynamic_sampler] == [len(batch) for batch in static_sampler]


def test_dynabatch_sampler_uses_precomputed_lengths_without_compute(monkeypatch):
    texts = ["a bb ccc", "dd eee ffff", "gg", "hhh iiii", "jjj k"]
    token_lengths = [3, 3, 1, 2, 2]
    word_lengths = [3, 3, 1, 2, 2]
    char_lengths = [8, 11, 2, 8, 5]

    def _boom(*args, **kwargs):
        raise AssertionError("compute_lengths should not be called when precomputed lengths are provided")

    monkeypatch.setattr(_main_module, "compute_lengths", _boom)
    sampler = dynabatch_sampler(
        texts=texts,
        tokenizer=object(),
        batch_size=2,
        token_lengths=token_lengths,
        word_lengths=word_lengths,
        char_lengths=char_lengths,
        dynamic_batch_mode=False,
    )
    batches = list(sampler)
    assert sum(len(batch) for batch in batches) == len(texts)


def test_dynabatch_sampler_precomputed_lengths_must_match_text_count(mock_tokenizer):
    texts = ["a", "b", "c"]
    with pytest.raises(ValueError, match="match `len\\(texts\\)`"):
        dynabatch_sampler(
            texts=texts,
            tokenizer=mock_tokenizer,
            batch_size=2,
            token_lengths=[1, 1],
            word_lengths=[1, 1, 1],
            char_lengths=[1, 1, 1],
        )


def test_batch_size_increased(sample_texts_5000, mock_word_tokenizer_session):
    from dynabatch.main import build_dynabatch_dataloader

    min_batch_size = 32
    loader = build_dynabatch_dataloader(
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
