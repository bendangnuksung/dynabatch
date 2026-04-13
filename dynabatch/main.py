import os
from functools import partial
from typing import Any, Callable

import pandas as pd
import torch
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset

from dynabatch.sampler import MaxTokenBatchSampler

# Avoid a hard dependency on transformers just for the type hint.
PreTrainedTokenizerBase = Any

# Prevent deadlocks with HuggingFace tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
) -> tuple[list[int], list[int], list[int], list[str]]:
    """
    Tokenize ``texts`` in parallel and return per-sequence length statistics.

    Returns a tuple of ``(token_lengths, word_lengths, char_lengths, truncated_texts)``
    where each list is aligned with the input ``texts``.  Sequences longer than
    ``max_length`` tokens are silently truncated and the shorter version is
    included in ``truncated_texts``.
    """

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
    return (
        df["token_lengths"].tolist(),
        df["word_lengths"].tolist(),
        df["char_lengths"].tolist(),
        df["texts"].tolist(),
    )


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
           first batch**. The prediction scale is centred around ``1.0``, meaning
           "same memory pressure as the first batch"; values above ``1.0`` signal
           higher pressure and OOM risk. The sampler picks the **largest** candidate
           whose prediction is ≤ ``threshold``. The default ``threshold`` of ``0.7``
           leaves a comfortable margin below the ``1.0`` reference point.

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
                                About ``1.0`` means "as memory-heavy as the first
                                batch"; above ``1.0`` means heavier (OOM risk).
                                Only candidates with prediction ≤ ``threshold`` are
                                kept; the largest kept size is used. Default ``0.7``
                                caps predicted load below the 1.0 reference. Lower =
                                more conservative; higher = larger batches, more risk.
        max_batch_range:        Maximum batch-size multiplier relative to ``batch_size``.
        shuffle:                If True, shuffle the order of the pre-built batches.
                                Sequences within each batch remain length-similar.
        shuffle_seed:           Seed for the RNG used to shuffle batches.
        shuffle_keep_first_n:   Number of leading batches kept in original order even
                                when shuffling, to ensure the first few batches always
                                fit within the established memory envelope.
        friendly_batch_size:    If True, round batch sizes to the nearest power of 2
                                (or 3 × power of 2). Recommended for training workloads.
        num_workers:            Number of parallel data-loading workers.
        debug:                  If True, disable parallel workers and print per-batch
                                sizing decisions. Use only during development.
        dynamic_batch_mode:     If False, all batches use exactly ``batch_size`` items
                                (equivalent to a standard MaxToken sampler).
        apply_template_func:    Optional function applied to the raw texts before
                                tokenization during collation.  Account for any extra
                                tokens it adds when setting ``max_input_token_length``.
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
    collate = partial(_collate_fn, tokenizer=tokenizer, apply_template_func=apply_template_func, **tokenizer_kwargs)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=num_workers)
