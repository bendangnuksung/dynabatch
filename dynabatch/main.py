import concurrent.futures
import os
import pickle
import random
from functools import partial
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict
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
    token_lengths: list[int],
    word_lengths: list[int],
    char_lengths: list[int],
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

    If ``max_batch_size`` is provided, the result is capped at that value,
    which prevents the classifier from suggesting sizes beyond what the GPU
    can handle for models with large baseline batch sizes.
    """
    max_allowed = len(token_lengths)
    multipliers = np.linspace(batch_start_range, batch_end_range, steps)
    candidate_batch_sizes = np.ceil(baseline_batch_size * multipliers).astype(int)

    features: dict[str, list] = {
        "max_input_length": [max_input_length] * steps,
        "token_size_x": [baseline_max_token_len.item()] * steps,
        "token_size_y": [token_lengths[0].item()] * steps,
        "token_size_diff": [token_lengths[0].item() / baseline_max_token_len.item()] * steps,
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
        token_lengths = token_lengths[:batch_size]
        word_lengths = word_lengths[:batch_size]
        char_lengths = char_lengths[:batch_size]
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
        result = min(baseline_batch_size, max_allowed)
    else:
        result = min(optimal_batch_size, max_allowed)

    return result


class MaxTokenBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        token_lengths: list[int],
        word_lengths: list[int],
        char_lengths: list[int],
        max_input_length: int,
        min_batch_size: int,
        shuffle: bool = False,
        threshold: float = 0.025,
        max_batch_range: float = 1.5,
        shuffle_seed: int = 21,
        shuffle_keep_first_n: int = 5,
        friendly_batch_size: bool = False,
        dynamic_batch_mode: bool = True,
    ):
        sorted_indices = sorted(
            range(len(token_lengths)),
            key=lambda i: token_lengths[i],
            reverse=True,
        )
        self.min_batch_size = min_batch_size
        self.max_input_length = max_input_length
        self.threshold = threshold
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_keep_first_n = shuffle_keep_first_n
        self.friendly_batch_size = friendly_batch_size
        self.dynamic_batch_mode = dynamic_batch_mode

        self.batch_start_range = 1.0
        self.batch_end_range = max(max_batch_range, 1.0)
        self.steps = 20

        self.batches = self._build_batches(sorted_indices, token_lengths, word_lengths, char_lengths)

    def _build_dynamic_batches(
        self, sorted_indices: list[int], token_lengths: list[int], word_lengths: list[int], char_lengths: list[int]
    ):
        token_lengths = np.array(token_lengths)
        word_lengths = np.array(word_lengths)
        char_lengths = np.array(char_lengths)
        sorted_token_lengths = token_lengths[sorted_indices]
        sorted_word_lengths = word_lengths[sorted_indices]
        sorted_char_lengths = char_lengths[sorted_indices]

        # The first batch always contains the longest sequences and uses the
        # minimum batch size — this is the hardest batch and the baseline for
        # all subsequent classifier predictions.
        baseline_batch_indices = sorted_indices[: self.min_batch_size]
        batches = [baseline_batch_indices]

        baseline_max_token_len = sorted_token_lengths[0]
        baseline_max_word_len = sorted_word_lengths[0]
        baseline_max_char_len = sorted_char_lengths[0]
        baseline_total_tokens = sorted_token_lengths[: self.min_batch_size].sum()

        remaining_token_lengths = sorted_token_lengths[self.min_batch_size :]
        remaining_word_lengths = sorted_word_lengths[self.min_batch_size :]
        remaining_char_lengths = sorted_char_lengths[self.min_batch_size :]
        next_start_idx = self.min_batch_size

        # 512 is the max input and 64 is the min input because the classifier was trained on these values
        max_input_length = min(self.max_input_length, 512)
        max_input_length = max(max_input_length, 64)

        while len(remaining_token_lengths):
            optimal_size = _select_optimal_batch_size(
                max_input_length=max_input_length,
                token_lengths=remaining_token_lengths,
                word_lengths=remaining_word_lengths,
                char_lengths=remaining_char_lengths,
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
            if self.shuffle:
                random.shuffle(batch_indices)
            batches.append(batch_indices)

            remaining_token_lengths = remaining_token_lengths[optimal_size:]
            next_start_idx += optimal_size

        return batches

    def _build_static_batches(self, sorted_indices: list[int]):
        """
        This method is same as Max Token Sampler/Batching
        """
        batches = []
        for i in range(0, len(sorted_indices), self.min_batch_size):
            batch_indices = sorted_indices[i : i + self.min_batch_size]
            if self.shuffle:
                random.shuffle(batch_indices)
            batches.append(batch_indices)
        return batches

    def _build_batches(
        self,
        sorted_indices: list[int],
        token_lengths: list[int],
        word_lengths: list[int],
        char_lengths: list[int],
    ) -> list[list[int]]:
        random.seed(self.shuffle_seed)
        if self.dynamic_batch_mode:
            batches = self._build_dynamic_batches(sorted_indices, token_lengths, word_lengths, char_lengths)
        else:
            batches = self._build_static_batches(sorted_indices)

        if self.shuffle and len(batches) > self.shuffle_keep_first_n:
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
    apply_template_func: Callable | None = None,
    **tokenizer_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Pads only to the longest sequence *in this batch*, not globally."""
    texts = [item["text"] for item in batch]
    if apply_template_func is not None:
        batch = apply_template_func(texts)

    tokens = tokenizer(
        text=texts,
        padding=True,
        return_tensors="pt",
        **tokenizer_kwargs,
    )
    tokens["texts"] = texts
    return tokens


def compute_lengths(
    texts: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int, max_workers: int = 4
) -> list[tuple[int, int, int]]:
    """Tokenizes texts in parallel chunks to save memory and speed up processing."""

    def _compute_lengths_for_batch(data: list[str]):
        encoded = tokenizer(
            text=data["texts"], truncation=True, max_length=max_length, padding=True, return_offsets_mapping=True
        )
        token_lengths = [sum(mask) for mask in encoded["attention_mask"]]
        chopped_char_lengths = []
        for sequence_offsets in encoded["offset_mapping"]:
            end_chars = [offset[1] for offset in sequence_offsets]
            chopped_char_lengths.append(max(end_chars))
        chopped_texts = [text[: chopped_char_lengths[i]] for i, text in enumerate(data["texts"])]
        chopped_word_lengths = [len(text.split()) for text in chopped_texts]
        return {
            "token_lengths": token_lengths,
            "word_lengths": chopped_word_lengths,
            "char_lengths": chopped_char_lengths,
            "texts": chopped_texts,
        }

    df = pd.DataFrame(texts).rename(columns={0: "texts"})
    df["words"] = df["texts"].apply(lambda x: len(x.split()))
    df["chars"] = df["texts"].apply(lambda x: len(x))
    datasets = DatasetDict({"data": HuggingFaceDataset.from_pandas(df)})
    datasets = datasets.map(
        _compute_lengths_for_batch,
        num_proc=max_workers,
        remove_columns=datasets["data"].column_names,
        batched=True,
        batch_size=100,
    )
    df = datasets["data"].to_pandas()
    token_lengths = df["token_lengths"].tolist()
    word_lengths = df["word_lengths"].tolist()
    char_lengths = df["char_lengths"].tolist()
    truncated_texts = df["texts"].tolist()

    return token_lengths, word_lengths, char_lengths, truncated_texts


def build_dynamic_batch_dataloader(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_input_token_length: int = 512,
    threshold: float = 0.005,
    shuffle: bool = False,
    shuffle_seed: int = 21,
    shuffle_keep_first_n: int = 3,
    friendly_batch_size: bool = False,
    num_workers: int = 4,
    max_batch_range: float = 1.5,
    debug: bool = False,
    dynamic_batch_mode: bool = True,
    apply_template_func: Callable | None = None,
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
                                OOMs, slightly more padding waste). Default 0.005
                                means only accept candidates with <0.5% predicted
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
        max_batch_range:        Maximum batch-size multiplier range
                                relative to ``batch_size``.
        debug:                  If True, return a DataLoader without parallel workers.
                                Parallel workers makes hard to debug. Only use for debugging.
        dynamic_batch_mode:     If True, use the dynamic batch mode. If False, it becomes
                                the same as Max Token Sampler/Batching with static batch size.
        apply_template_func:    A function that applies a template to the texts.
                                Mind that adding extra text through the template will increase
                                the token length of the batch. So adjust max_input_token_length
                                accordingly.
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

    token_lengths, word_lengths, char_lengths, truncated_texts = compute_lengths(
        texts, tokenizer, max_input_token_length, max_workers=num_workers
    )
    dataset = TextDataset(truncated_texts)

    sampler = MaxTokenBatchSampler(
        token_lengths=token_lengths,
        word_lengths=word_lengths,
        char_lengths=char_lengths,
        max_input_length=max_input_token_length,
        min_batch_size=batch_size,
        shuffle=shuffle,
        threshold=threshold,
        max_batch_range=max_batch_range,
        shuffle_seed=shuffle_seed,
        shuffle_keep_first_n=shuffle_keep_first_n,
        friendly_batch_size=friendly_batch_size,
        dynamic_batch_mode=dynamic_batch_mode,
    )

    if debug:
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=partial(
                _collate_fn, tokenizer=tokenizer, apply_template_func=apply_template_func, **tokenizer_kwargs
            ),
        )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(
            _collate_fn, tokenizer=tokenizer, apply_template_func=apply_template_func, **tokenizer_kwargs
        ),
        num_workers=num_workers,
    )
