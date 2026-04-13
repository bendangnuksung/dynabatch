import os
import random
from functools import partial
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from dynabatch.utils import get_hardware_friendly_batch_size

# Avoid a hard dependency on transformers just for the type hint.
PreTrainedTokenizerBase = Any

# Prevent deadlocks with Hugging Face tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


_REGRESSOR_PATH = os.path.join(os.path.dirname(__file__), "models", "regressor.ubj")
_REGRESSOR = xgb.XGBRegressor()
_REGRESSOR.load_model(_REGRESSOR_PATH)


def _select_optimal_batch_size(
    token_lengths: list[int],
    word_lengths: list[int],
    char_lengths: list[int],
    baseline_features: dict[str, Any],
    threshold: float,
    candidate_batch_sizes: list[int],
) -> int:
    """
    Use the pre-trained regressor to find the largest batch size whose predicted
    memory pressure stays at or below ``threshold`` relative to the first batch.

    Generates candidate batch sizes between
    ``baseline_batch_size * batch_start_range`` and
    ``baseline_batch_size * batch_end_range``, builds the same feature set the
    regressor was trained on, and returns the largest candidate whose prediction
    is at or below ``threshold`` (see ``build_dynamic_batch_dataloader`` docs for
    how that scale relates to the baseline batch).
    """
    features: dict[str, list] = {
        "batch_size_y": [],
        "token_mean_y": [],
        "token_std_y": [],
        "token_sum_y": [],
        "token_max_y": [],
        # All commented out feature are not used, keeping them for future if we want to use them.
        # "token_min_y": [],
        # "token_median_y": [],
        # "token_mode_y": [],
        "word_mean_y": [],
        "word_sum_y": [],
        "word_max_y": [],
        # "word_std_y": [],
        # "word_min_y": [],
        # "word_median_y": [],
        # "word_mode_y": [],
        # "char_mean_y": [],
        # "char_std_y": [],
        # "char_min_y": [],
        # "char_median_y": [],
        # "char_mode_y": [],
        # "char_max_y": [],
        "char_sum_y": [],
    }

    for batch_size in candidate_batch_sizes:
        tl = token_lengths[:batch_size]
        wl = word_lengths[:batch_size]
        cl = char_lengths[:batch_size]
        features["batch_size_y"].append(batch_size)

        features["token_mean_y"].append(tl.mean())
        features["token_std_y"].append(tl.std())
        features["token_sum_y"].append(tl.sum())
        features["token_max_y"].append(tl.max())
        # All commented out feature are not used, keeping them for future if we want to use them.
        # features["token_min_y"].append(tl.min())
        # features["token_median_y"].append(np.median(tl).astype(int))
        # features["token_mode_y"].append(np.bincount(tl).argmax())

        features["word_mean_y"].append(wl.mean())
        features["word_sum_y"].append(wl.sum())
        features["word_max_y"].append(wl.max())
        # features["word_std_y"].append(wl.std())
        # features["word_min_y"].append(wl.min())
        # features["word_median_y"].append(np.median(wl).astype(int))
        # features["word_mode_y"].append(np.bincount(wl).argmax())

        # features["char_mean_y"].append(cl.mean())
        # features["char_std_y"].append(cl.std())
        # features["char_min_y"].append(cl.min())
        # features["char_median_y"].append(np.median(cl).astype(int))
        # features["char_mode_y"].append(np.bincount(cl).argmax())
        # features["char_max_y"].append(cl.max())
        features["char_sum_y"].append(cl.sum())

    features.update(baseline_features)
    feature_df = pd.DataFrame(features)

    # get diffs features
    feature_df["batch_size_diff"] = feature_df["batch_size_y"] / feature_df["batch_size_x"]
    feature_df["token_max_diff"] = feature_df["token_max_y"] / feature_df["token_max_x"]
    feature_df["token_mean_diff"] = feature_df["token_mean_y"] / feature_df["token_mean_x"]
    feature_df["token_sum_diff"] = feature_df["token_sum_y"] / feature_df["token_sum_x"]

    feature_df["word_max_diff"] = feature_df["word_max_y"] / feature_df["word_max_x"]
    feature_df["word_mean_diff"] = feature_df["word_mean_y"] / feature_df["word_mean_x"]
    feature_df["word_sum_diff"] = feature_df["word_sum_y"] / feature_df["word_sum_x"]

    feature_df["char_sum_diff"] = feature_df["char_sum_y"] / feature_df["char_sum_x"]

    feature_df_selected = feature_df[_REGRESSOR.get_booster().feature_names]

    preds_raw = _REGRESSOR.predict(feature_df_selected)
    preds = (preds_raw <= threshold).astype(int)
    optimal_batch_size = int(np.max(preds * candidate_batch_sizes))

    if optimal_batch_size == 0:
        return baseline_features["batch_size_x"][0], feature_df
    return optimal_batch_size, feature_df


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
        debug: bool = False,
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
        self.steps = int((self.batch_end_range - self.batch_start_range) * 20)

        self.debug = debug

        self.batches = self._build_batches(sorted_indices, token_lengths, word_lengths, char_lengths)

    def _build_dynamic_batches(
        self, sorted_indices: list[int], token_lengths: list[int], word_lengths: list[int], char_lengths: list[int]
    ):
        sorted_token_lengths = np.array(token_lengths)[sorted_indices]
        sorted_word_lengths = np.array(word_lengths)[sorted_indices]
        sorted_char_lengths = np.array(char_lengths)[sorted_indices]

        multipliers = np.linspace(self.batch_start_range, self.batch_end_range, self.steps)
        candidate_batch_sizes = np.round(self.min_batch_size * multipliers).astype(int)
        candidate_batch_sizes = np.unique(candidate_batch_sizes)

        # The first batch always contains the longest sequences and uses the
        # minimum batch size — this is the hardest batch and the baseline for
        # all subsequent regressor predictions.
        baseline_batch_indices = sorted_indices[: self.min_batch_size]
        batches = [baseline_batch_indices]

        baseline_features = {
            "batch_size_x": [self.min_batch_size] * len(candidate_batch_sizes),
            "token_mean_x": [sorted_token_lengths[: self.min_batch_size].mean()] * len(candidate_batch_sizes),
            "token_sum_x": [sorted_token_lengths[: self.min_batch_size].sum()] * len(candidate_batch_sizes),
            "token_max_x": [sorted_token_lengths[: self.min_batch_size].max()] * len(candidate_batch_sizes),
            # All commented out feature are not used, keeping them for future if we want to use them.
            # "token_std_x": [sorted_token_lengths[: self.min_batch_size].std()] * len(candidate_batch_sizes),
            # "token_min_x": [sorted_token_lengths[: self.min_batch_size].min()] * len(candidate_batch_sizes),
            # "token_median_x": [np.median(sorted_token_lengths[: self.min_batch_size]).astype(int)]
            # * len(candidate_batch_sizes),
            # "token_mode_x": [np.bincount(sorted_token_lengths[: self.min_batch_size]).argmax()]
            # * len(candidate_batch_sizes),
            "word_mean_x": [sorted_word_lengths[: self.min_batch_size].mean()] * len(candidate_batch_sizes),
            "word_sum_x": [sorted_word_lengths[: self.min_batch_size].sum()] * len(candidate_batch_sizes),
            "word_max_x": [sorted_word_lengths[: self.min_batch_size].max()] * len(candidate_batch_sizes),
            # "word_std_x": [sorted_word_lengths[: self.min_batch_size].std()] * len(candidate_batch_sizes),
            # "word_min_x": [sorted_word_lengths[: self.min_batch_size].min()] * len(candidate_batch_sizes),
            # "word_median_x": [np.median(sorted_word_lengths[: self.min_batch_size]).astype(int)] * len(candidate_batch_sizes),
            # "word_mode_x": [np.bincount(sorted_word_lengths[: self.min_batch_size]).argmax()] * len(candidate_batch_sizes),
            "char_sum_x": [sorted_char_lengths[: self.min_batch_size].sum()] * len(candidate_batch_sizes),
            # "char_mean_x": [sorted_char_lengths[: self.min_batch_size].mean()] * len(candidate_batch_sizes),
            # "char_std_x": [sorted_char_lengths[: self.min_batch_size].std()] * len(candidate_batch_sizes),
            # "char_min_x": [sorted_char_lengths[: self.min_batch_size].min()] * len(candidate_batch_sizes),
            # "char_median_x": [np.median(sorted_char_lengths[: self.min_batch_size]).astype(int)] * len(candidate_batch_sizes),
            # "char_mode_x": [np.bincount(sorted_char_lengths[: self.min_batch_size]).argmax()] * len(candidate_batch_sizes),
            # "char_max_x": [sorted_char_lengths[: self.min_batch_size].max()] * len(candidate_batch_sizes),
        }

        remaining_token_lengths = sorted_token_lengths[self.min_batch_size :]
        remaining_word_lengths = sorted_word_lengths[self.min_batch_size :]
        remaining_char_lengths = sorted_char_lengths[self.min_batch_size :]
        next_start_idx = self.min_batch_size

        total_remaining = len(remaining_token_lengths)

        with tqdm(
            total=total_remaining,
            desc="Step 2: building dynamic batches",
            unit="seq",
        ) as pbar:
            while len(remaining_token_lengths):
                optimal_size, feature_df = _select_optimal_batch_size(
                    token_lengths=remaining_token_lengths,
                    word_lengths=remaining_word_lengths,
                    char_lengths=remaining_char_lengths,
                    baseline_features=baseline_features,
                    threshold=self.threshold,
                    candidate_batch_sizes=candidate_batch_sizes,
                )
                if self.friendly_batch_size:
                    optimal_size = get_hardware_friendly_batch_size(optimal_size)

                # is_last
                if len(remaining_token_lengths) <= optimal_size:
                    optimal_size = len(remaining_token_lengths)
                batch_indices = [sorted_indices[next_start_idx + i] for i in range(optimal_size)]
                if self.shuffle:
                    random.shuffle(batch_indices)
                batches.append(batch_indices)

                remaining_token_lengths = remaining_token_lengths[optimal_size:]
                remaining_word_lengths = remaining_word_lengths[optimal_size:]
                remaining_char_lengths = remaining_char_lengths[optimal_size:]
                next_start_idx += optimal_size
                pbar.update(optimal_size)

                if self.debug:
                    print(f"Batch N: {len(batches)} \t|\t Batch Size: {optimal_size}")
                    print("-" * 40)

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
        desc="Step 1: tokenizing and measuring lengths",
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
    threshold: float = 0.7,
    max_batch_range: float = 2.0,
    shuffle: bool = False,
    shuffle_seed: int = 21,
    shuffle_keep_first_n: int = 3,
    friendly_batch_size: bool = False,
    num_workers: int = 4,
    debug: bool = False,
    dynamic_batch_mode: bool = True,
    apply_template_func: Callable | None = None,
    **tokenizer_kwargs: Any,
) -> DataLoader:
    """
    A drop-in replacement for a standard DataLoader that eliminates padding waste
    by using a pre-trained regressor to dynamically size each batch while staying
    near the memory envelope established by the first batch.

    How it works:
        1. All sequences are sorted by token length, longest first.
        2. The first batch always uses exactly ``batch_size`` items — the hardest
           batch (longest sequences). Choose ``batch_size`` as the largest count
           that does not OOM on those worst-case inputs; the regressor treats that
           batch as the memory baseline for all later sizing decisions.
        3. For every later batch, an XGBRegressor (trained offline on real GPU
           memory profiles) scores each candidate batch size **relative to that
           first batch**. Interpret the prediction on a scale where about ``1.0``
           means "similar memory pressure to the first batch"; values above
           ``1.0`` mean heavier than that baseline and point to OOM risk. The
           sampler picks the **largest** candidate whose prediction is
           ≤ ``threshold``. The default ``threshold`` of ``0.75`` stays below the
           ``1.0`` reference, leaving headroom versus the max-safe first batch.

        The regressor was trained on empirical data: actual GPU memory usage
        recorded across many (model, batch_size, max_length) combinations. It
        learned the relationship between token counts, padding, batch size ratios,
        and GPU memory pressure — so it generalises across models and hardware
        without per-setup calibration.

    Args:
        texts:                  Raw input strings of any length.
        tokenizer:              Any HuggingFace tokenizer.
        batch_size:             Batch size for the first (hardest) batch: the maximum
                                number of longest sequences your GPU can run without
                                OOM. This fixes the memory baseline the regressor
                                compares against; it may increase counts on later,
                                shorter-sequence batches subject to ``threshold``.
        max_input_token_length: Hard truncation limit per sequence. Sequences longer
                                than this are silently truncated.
        threshold:              Maximum allowed regressor prediction when comparing a
                                candidate batch to the first ``batch_size`` batch.
                                The model is trained so that about ``1.0`` means
                                "as memory-heavy as that first batch"; values above
                                ``1.0`` mean heavier than the baseline (OOM risk).
                                Only candidates with prediction ≤ ``threshold`` are
                                kept; the largest kept size is used. Default ``0.75``
                                caps predicted load below that 1.0 reference. Lower =
                                more conservative; higher = larger batches, more risk.
        max_batch_range:        Maximum batch-size multiplier range
                                relative to ``batch_size``.
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
        debug=debug,
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
