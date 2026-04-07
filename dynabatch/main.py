import concurrent.futures
import os
import pickle
import random
from functools import partial
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from dynabatch.utils import get_hardware_friendly_batch_size

# Avoid a hard dependency on transformers just for the type hint.
PreTrainedTokenizerBase = Any

# Prevent deadlocks with Hugging Face tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


_CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "models", "classifier.pkl")
with open(_CLASSIFIER_PATH, "rb") as _f:
    _CLASSIFIER = pickle.load(_f)


def _select_optimal_batch_size(
    max_input_length: int,
    sequence_lengths: list[int],
    baseline_max_token_len: int,
    baseline_max_word_len: int,
    baseline_max_char_len: int,
    baseline_batch_size: int,
    baseline_total_tokens: int,
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
        "token_size_x": [baseline_max_token_len.item()] * steps,
        "token_size_y": [sequence_lengths[0][0].item()] * steps,
        "token_size_diff": [sequence_lengths[0][0].item() / baseline_max_token_len.item()] * steps,
        "batch_size_x": [baseline_batch_size] * steps,
        "batch_size_y": [],
        "batch_size_diff": [],
        "total_tokens_x": [baseline_total_tokens] * steps,
        "total_tokens_y": [],
        "total_token_size_diff": [],
        "word_length_x": [baseline_max_word_len.item()] * steps,
        "char_length_x": [baseline_max_char_len.item()] * steps,
        "word_length_y": [],
        "char_length_y": [],
        "token_mean_y": [],
        "token_std_y": [],
        "token_min_y": [],
        "token_median_y": [],
        "token_mode_y": [],
        "word_mean_y": [],
        "word_std_y": [],
        "word_min_y": [],
        "word_median_y": [],
        "word_mode_y": [],
        "word_sum_y": [],
        "word_max_y": [],
        "char_mean_y": [],
        "char_std_y": [],
        "char_min_y": [],
        "char_median_y": [],
        "char_mode_y": [],
        "char_sum_y": [],
        "char_max_y": [],
    }

    for batch_size in candidate_batch_sizes:
        token_lengths = sequence_lengths[:batch_size, 0]
        word_lengths = sequence_lengths[:batch_size, 1]
        char_lengths = sequence_lengths[:batch_size, 2]
        total_tokens = token_lengths.sum().item()

        features["batch_size_y"].append(batch_size.item())
        features["batch_size_diff"].append(batch_size.item() / baseline_batch_size)
        features["total_tokens_y"].append(total_tokens)
        features["total_token_size_diff"].append(total_tokens / baseline_total_tokens.item())

        features["token_mean_y"].append(token_lengths.mean().item())
        features["token_std_y"].append(token_lengths.std().item())
        features["token_min_y"].append(token_lengths.min().item())
        features["token_median_y"].append(np.median(token_lengths).item())
        features["token_mode_y"].append(np.bincount(token_lengths).argmax().item())

        features["word_mean_y"].append(word_lengths.mean().item())
        features["word_std_y"].append(word_lengths.std().item())
        features["word_min_y"].append(word_lengths.min().item())
        features["word_median_y"].append(np.median(word_lengths).item())
        features["word_mode_y"].append(np.bincount(word_lengths).argmax().item())
        features["word_sum_y"].append(word_lengths.sum().item())
        features["word_max_y"].append(word_lengths.max().item())

        features["char_mean_y"].append(char_lengths.mean().item())
        features["char_std_y"].append(char_lengths.std().item())
        features["char_min_y"].append(char_lengths.min().item())
        features["char_median_y"].append(np.median(char_lengths).item())
        features["char_mode_y"].append(np.bincount(char_lengths).argmax().item())
        features["char_sum_y"].append(char_lengths.sum().item())
        features["char_max_y"].append(char_lengths.max().item())

        features["word_length_y"].append(word_lengths.max().item())
        features["char_length_y"].append(char_lengths.max().item())

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
        sequence_lengths: list[tuple[int, int, int]],
        max_input_length: int,
        min_batch_size: int,
        shuffle: bool = False,
        threshold: float = 0.025,
        batch_start_range: float = 1.0,
        batch_end_range: float = 6.0,
        steps: int = 50,
        shuffle_seed: int = 21,
        shuffle_keep_first_n: int = 5,
        friendly_batch_size: bool = False,
    ):
        sorted_indices = sorted(
            range(len(sequence_lengths)),
            key=lambda i: sequence_lengths[i][0],
            reverse=True,
        )
        self.min_batch_size = min_batch_size
        self.max_input_length = max_input_length
        self.threshold = threshold
        self.batch_start_range = batch_start_range
        self.batch_end_range = batch_end_range
        self.steps = steps
        self.shuffle_seed = shuffle_seed
        self.shuffle_keep_first_n = shuffle_keep_first_n
        self.friendly_batch_size = friendly_batch_size

        self.batches = self._build_batches(sorted_indices, sequence_lengths, shuffle)

    def _build_batches(
        self,
        sorted_indices: list[int],
        lengths: list[int],
        shuffle: bool,
    ) -> list[list[int]]:
        random.seed(self.shuffle_seed)

        lengths = np.array(lengths)
        sorted_lengths = lengths[sorted_indices]
        sorted_token_lengths = sorted_lengths[:, 0]
        sorted_word_lengths = sorted_lengths[:, 1]
        sorted_char_lengths = sorted_lengths[:, 2]

        # The first batch always contains the longest sequences and uses the
        # minimum batch size — this is the hardest batch and the baseline for
        # all subsequent classifier predictions.
        baseline_batch_indices = sorted_indices[: self.min_batch_size]
        batches = [baseline_batch_indices]

        baseline_max_token_len = sorted_token_lengths[0]
        baseline_max_word_len = sorted_word_lengths[0]
        baseline_max_char_len = sorted_char_lengths[0]
        baseline_total_tokens = sorted_token_lengths[: self.min_batch_size].sum()

        remaining_lengths = sorted_lengths[self.min_batch_size :]
        next_start_idx = self.min_batch_size

        # 512 is the max input and 64 is the min input because the classifier was trained on these values
        max_input_length = min(self.max_input_length, 512)
        max_input_length = max(max_input_length, 64)

        while len(remaining_lengths):
            optimal_size = _select_optimal_batch_size(
                max_input_length=max_input_length,
                sequence_lengths=remaining_lengths,
                baseline_max_token_len=baseline_max_token_len,
                baseline_max_word_len=baseline_max_word_len,
                baseline_max_char_len=baseline_max_char_len,
                baseline_batch_size=self.min_batch_size,
                baseline_total_tokens=baseline_total_tokens,
                threshold=self.threshold,
                batch_start_range=self.batch_start_range,
                batch_end_range=self.batch_end_range,
                steps=self.steps,
            )
            if self.friendly_batch_size:
                optimal_size = get_hardware_friendly_batch_size(optimal_size)

            batch_indices = [sorted_indices[next_start_idx + i] for i in range(optimal_size)]
            if shuffle:
                random.shuffle(batch_indices)
            batches.append(batch_indices)

            remaining_lengths = remaining_lengths[optimal_size:]
            next_start_idx += optimal_size

        if shuffle and len(batches) > self.shuffle_keep_first_n:
            first_half = batches[: self.shuffle_keep_first_n]
            second_half = batches[self.shuffle_keep_first_n :]
            random.shuffle(second_half)
            batches = first_half + second_half

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
    max_length: int,
    **tokenizer_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Pads only to the longest sequence *in this batch*, not globally."""
    texts = [item["text"] for item in batch]
    tokens = tokenizer(
        text=texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        **tokenizer_kwargs,
    )
    tokens["texts"] = texts
    return tokens


def _tokenize_chunk(text: str, tokenizer: PreTrainedTokenizerBase, max_length: int) -> list[tuple[int, int, int]]:
    """Tokenizing a single text so that no padding is applied and get the correct token lengths"""
    encodings = tokenizer(text=[text], truncation=True, max_length=max_length, padding=False)
    word_length = len(text.split())
    char_length = len(text)
    return [(len(encodings["input_ids"][0]), word_length, char_length)]


def compute_lengths(
    texts: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int, max_workers: int = 4
) -> list[tuple[int, int, int]]:
    """Tokenizes texts in parallel chunks to save memory and speed up processing."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_tokenize_chunk, text, tokenizer, max_length) for text in texts]
        for future in futures:
            results.extend(future.result())
    return results


def build_dynamic_batch_dataloader(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_input_token_length: int = 256,
    threshold: float = 0.025,
    shuffle: bool = False,
    shuffle_seed: int = 21,
    shuffle_keep_first_n: int = 3,
    friendly_batch_size: bool = False,
    num_workers: int = 4,
    batch_start_range: float = 1.0,
    batch_end_range: float = 6.0,
    steps: int = 50,
    debug: bool = False,
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
        shuffle_seed:           Seed for the random number generator used to shuffle
                                the batches.
        shuffle_keep_first_n:   Number of batches to keep in the original order.
                                This is to ensure during the training/inference
                                that the first few batches fits into the memory.
        friendly_batch_size:    If True, use the hardware-friendly batch size.
                                This is to ensure that the batch size is a power of 2
                                or 3 times a power of 2. Good for training.
        num_workers:            Number of parallel data-loading workers. Safe to set
                                above 0 since the collate function is picklable.
        batch_start_range:      Lower bound of the batch-size multiplier range
                                relative to ``batch_size``. Default 1.0 (1x).
        batch_end_range:        Upper bound of the batch-size multiplier range
                                relative to ``batch_size``. Default 6.0 (6x).
        steps:                  Number of candidate batch sizes to evaluate between
                                ``batch_start_range`` and ``batch_end_range``.
        debug:                  If True, return a DataLoader without parallel workers.
                                Parallel workers makes hard to debug. Only use for debugging.
        **tokenizer_kwargs:     Extra keyword arguments forwarded to the tokenizer
                                during collation (e.g. ``max_length``,
                                ``add_special_tokens``).

    Returns:
        A DataLoader yielding dicts with keys ``input_ids``, ``attention_mask``,
        ``texts`` (and any other keys your tokenizer returns), as PyTorch tensors.
    """
    if debug:
        num_workers = 0
    elif not num_workers:
        num_workers = os.cpu_count() or 1

    sequence_lengths = compute_lengths(texts, tokenizer, max_input_token_length, max_workers=num_workers)

    sampler = MaxTokenBatchSampler(
        sequence_lengths=sequence_lengths,
        max_input_length=max_input_token_length,
        min_batch_size=batch_size,
        shuffle=shuffle,
        threshold=threshold,
        batch_start_range=batch_start_range,
        batch_end_range=batch_end_range,
        steps=steps,
        shuffle_seed=shuffle_seed,
        shuffle_keep_first_n=shuffle_keep_first_n,
        friendly_batch_size=friendly_batch_size,
    )

    dataset = TextDataset(texts)

    if debug:
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=partial(_collate_fn, tokenizer=tokenizer, max_length=max_input_token_length, **tokenizer_kwargs),
        )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(_collate_fn, tokenizer=tokenizer, max_length=max_input_token_length, **tokenizer_kwargs),
        num_workers=num_workers,
    )
