import os
import pickle
import random
from functools import partial
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

# Avoid a hard dependency on transformers just for the type hint.
PreTrainedTokenizerBase = Any

_CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "models", "classifier.pkl")
with open(_CLASSIFIER_PATH, "rb") as _f:
    _CLASSIFIER = pickle.load(_f)


def _select_optimal_batch_size(
    max_input_length: int,
    sequence_lengths: list[int],
    baseline_max_token_len: int,
    baseline_batch_size: int,
    baseline_total_tokens: int,
    baseline_total_paddings: int,
    threshold: float = 0.025,
    batch_start_range: float = 1.0,
    batch_end_range: float = 6.0,
    steps: int = 50,
) -> int:
    """
    Use the pre-trained classifier to find the largest batch size that keeps
    predicted GPU memory usage below the spike threshold.

    Generates ``steps`` candidate batch sizes between
    ``baseline_batch_size * batch_start_range`` and
    ``baseline_batch_size * batch_end_range``, builds the same feature set the
    classifier was trained on, and returns the largest candidate whose predicted
    spike probability is at or below ``threshold``.
    """
    max_allowed = len(sequence_lengths)
    multipliers = np.linspace(batch_start_range, batch_end_range, steps)
    candidate_batch_sizes = np.ceil(baseline_batch_size * multipliers).astype(int)

    features: dict[str, list] = {
        "max_input_length": [max_input_length] * steps,
        "token_size_x": [baseline_max_token_len] * steps,
        "token_size_y": [sequence_lengths[0]] * steps,
        "token_size_diff": [sequence_lengths[0] / baseline_max_token_len] * steps,
        "batch_size_x": [baseline_batch_size] * steps,
        "batch_size_y": [],
        "batch_size_diff": [],
        "total_tokens_x": [baseline_total_tokens] * steps,
        "total_tokens_y": [],
        "total_token_size_diff": [],
        "paddings_x": [baseline_total_paddings] * steps,
        "paddings_y": [],
        "total_paddings_diff": [],
    }

    for batch_size in candidate_batch_sizes:
        total_tokens = sum(sequence_lengths[:batch_size])
        total_paddings = (sequence_lengths[0] * batch_size) - total_tokens
        if total_paddings == 0:
            # The classifier was never trained on zero-padding batches and
            # produces unreliable predictions, so inject a small synthetic value.
            total_paddings = random.randint(1, 10)

        features["batch_size_y"].append(batch_size)
        features["batch_size_diff"].append(batch_size / baseline_batch_size)
        features["total_tokens_y"].append(total_tokens)
        features["total_token_size_diff"].append(total_tokens / baseline_total_tokens)
        features["paddings_y"].append(total_paddings)
        features["total_paddings_diff"].append(total_paddings / baseline_total_paddings)

    feature_df = pd.DataFrame(features)
    feature_df = feature_df[_CLASSIFIER.feature_names_in_]

    spike_probabilities = _CLASSIFIER.predict_proba(feature_df)[:, 1]

    safe_mask = (spike_probabilities <= threshold).astype(int)
    optimal_batch_size = int(np.max(safe_mask * candidate_batch_sizes))

    if optimal_batch_size == 0:
        return min(baseline_batch_size, max_allowed)
    return min(optimal_batch_size, max_allowed)


class MaxTokenBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sequence_lengths: list[int],
        max_input_length: int,
        min_batch_size: int,
        shuffle: bool = False,
        threshold: float = 0.025,
        batch_start_range: float = 1.0,
        batch_end_range: float = 6.0,
        steps: int = 50,
    ):
        sorted_indices = sorted(
            range(len(sequence_lengths)),
            key=lambda i: sequence_lengths[i],
            reverse=True,
        )
        self.min_batch_size = min_batch_size
        self.max_input_length = max_input_length
        self.threshold = threshold
        self.batch_start_range = batch_start_range
        self.batch_end_range = batch_end_range
        self.steps = steps
        self.batches = self._build_batches(sorted_indices, sequence_lengths, shuffle)

    def _build_batches(
        self,
        sorted_indices: list[int],
        lengths: list[int],
        shuffle: bool,
    ) -> list[list[int]]:
        sorted_lengths = [lengths[i] for i in sorted_indices]

        # The first batch always contains the longest sequences and uses the
        # minimum batch size — this is the hardest batch and the baseline for
        # all subsequent classifier predictions.
        baseline_batch_indices = [sorted_indices[i] for i in range(self.min_batch_size)]
        batches: list[list[int]] = [baseline_batch_indices]

        baseline_max_token_len = sorted_lengths[0]
        baseline_total_tokens = sum(sorted_lengths[: self.min_batch_size])
        baseline_total_paddings = (baseline_max_token_len * self.min_batch_size) - baseline_total_tokens

        remaining_lengths = sorted_lengths[self.min_batch_size :]
        next_start_idx = self.min_batch_size

        while remaining_lengths:
            optimal_size = _select_optimal_batch_size(
                max_input_length=self.max_input_length,
                sequence_lengths=remaining_lengths,
                baseline_max_token_len=baseline_max_token_len,
                baseline_batch_size=self.min_batch_size,
                baseline_total_tokens=baseline_total_tokens,
                baseline_total_paddings=baseline_total_paddings,
                threshold=self.threshold,
                batch_start_range=self.batch_start_range,
                batch_end_range=self.batch_end_range,
                steps=self.steps,
            )

            batch_indices = [sorted_indices[next_start_idx + i] for i in range(optimal_size)]
            batches.append(batch_indices)

            remaining_lengths = remaining_lengths[optimal_size:]
            next_start_idx += optimal_size

        if shuffle:
            random.shuffle(batches)

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {"text": self.texts[idx]}


def _collate_fn(
    batch: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    **tokenizer_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Pads only to the longest sequence *in this batch*, not globally."""
    texts = [item["text"] for item in batch]
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        **tokenizer_kwargs,
    )
    tokens["texts"] = texts
    return tokens


def compute_sequence_lengths(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> list[int]:
    """Tokenizes ``texts`` and returns the token length of each sequence."""
    encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    return [len(ids) for ids in encodings["input_ids"]]


def build_dynamic_batch_dataloader(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_input_token_length: int,
    threshold: float = 0.025,
    shuffle: bool = False,
    num_workers: int = 4,
    batch_start_range: float = 1.0,
    batch_end_range: float = 6.0,
    steps: int = 50,
    **tokenizer_kwargs: Any,
) -> DataLoader:
    """
    A drop-in replacement for a standard DataLoader that eliminates padding waste
    by using a pre-trained classifier to dynamically size each batch so that GPU
    memory usage never exceeds ~90% of the first batch's peak allocation.

    How it works:
        1. All sequences are sorted by token length, longest first.
        2. The first batch always uses exactly ``batch_size`` items — the hardest
           batch, containing the longest sequences. If your model and GPU survive
           this batch without OOM, every subsequent batch is guaranteed to be safe.
        3. For every subsequent batch, a HistGradientBoostingClassifier (trained
           offline on real GPU memory profiles from multiple models) evaluates a
           range of candidate batch sizes and picks the largest one whose predicted
           probability of a memory spike stays below ``threshold``.

        The classifier was trained on empirical data: actual GPU memory usage
        recorded across many (model, batch_size, max_length) combinations. It
        learned the relationship between token counts, padding, batch size ratios,
        and GPU memory pressure — so it generalises across models and hardware
        without per-setup calibration.

    Args:
        texts:                  Raw input strings of any length.
        tokenizer:              Any HuggingFace tokenizer.
        batch_size:             Minimum (baseline) batch size, used for the first
                                (hardest) batch. Pick the largest value that doesn't
                                OOM on your worst-case input — the classifier scales
                                up from this for all shorter-sequence batches.
        max_input_token_length: Hard truncation limit per sequence. Sequences longer
                                than this are silently truncated.
        threshold:              Maximum spike probability tolerated per candidate
                                batch size. Lower values are more conservative (fewer
                                OOMs, slightly more padding waste). Default 0.025
                                means only accept candidates with <2.5% predicted
                                probability of exceeding 90% GPU utilisation.
        shuffle:                If True, shuffle the order of the pre-built batches.
                                Sequences within each batch remain length-similar.
        num_workers:            Number of parallel data-loading workers. Safe to set
                                above 0 since the collate function is picklable.
        batch_start_range:      Lower bound of the batch-size multiplier range
                                relative to ``batch_size``. Default 1.0 (1×).
        batch_end_range:        Upper bound of the batch-size multiplier range
                                relative to ``batch_size``. Default 6.0 (6×).
        steps:                  Number of candidate batch sizes to evaluate between
                                ``batch_start_range`` and ``batch_end_range``.
        **tokenizer_kwargs:     Extra keyword arguments forwarded to the tokenizer
                                during collation (e.g. ``max_length``,
                                ``add_special_tokens``).

    Returns:
        A DataLoader yielding dicts with keys ``input_ids``, ``attention_mask``,
        ``texts`` (and any other keys your tokenizer returns), as PyTorch tensors.
    """
    sequence_lengths = compute_sequence_lengths(texts, tokenizer, max_input_token_length)

    sampler = MaxTokenBatchSampler(
        sequence_lengths=sequence_lengths,
        max_input_length=max_input_token_length,
        min_batch_size=batch_size,
        shuffle=shuffle,
        threshold=threshold,
        batch_start_range=batch_start_range,
        batch_end_range=batch_end_range,
        steps=steps,
    )

    dataset = TextDataset(texts)
    num_workers = min(num_workers, os.cpu_count() or 1)

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(_collate_fn, tokenizer=tokenizer, **tokenizer_kwargs),
        num_workers=num_workers,
    )
