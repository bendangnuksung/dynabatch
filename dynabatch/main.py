import os
from functools import partial
from typing import Any

import torch
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import DataLoader, Dataset

from dynabatch.sampler import DynaBatchSampler

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
    **tokenizer_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Pads only to the longest sequence *in this batch*, not globally."""
    texts = [item["text"] for item in batch]
    tokens = tokenizer(
        text=texts,
        padding=True,
        return_tensors="pt",
        **tokenizer_kwargs,
    )
    tokens["texts"] = texts
    return tokens


def _supports_offset_mapping(tokenizer: PreTrainedTokenizerBase) -> bool:
    if not getattr(tokenizer, "is_fast", False):
        return False
    try:
        tokenizer("hello world", return_offsets_mapping=True)
        return True
    except (NotImplementedError, ValueError, TypeError):
        return False


def _align_decoded_to_original(original: str, decoded: str) -> int:
    n = len(decoded.rstrip())
    if n == 0:
        return 0
    end = min(n, len(original))
    while end < len(original) and original[end].isspace():
        end += 1
    return end


def _compute_char_len_from_offsets(offsets: list[tuple[int, int]]) -> int:
    return max((end for _, end in offsets if end != 0), default=0)


def _batch_decode_cached_specials(input_ids: list[list[int]], tokenizer: PreTrainedTokenizerBase) -> list[str]:
    convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    convert_tokens_to_string = getattr(tokenizer, "convert_tokens_to_string", None)
    if not callable(convert_ids_to_tokens) or not callable(convert_tokens_to_string):
        return tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    decoded_texts: list[str] = []
    append_decoded_text = decoded_texts.append

    for ids in input_ids:
        filtered_ids = [token_id for token_id in ids if token_id not in special_ids]
        tokens = convert_ids_to_tokens(filtered_ids, skip_special_tokens=False)
        append_decoded_text(convert_tokens_to_string(tokens))

    return decoded_texts


def _process_chunk_hf_offsets(
    chunk: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> tuple[list[int], list[int], list[int], list[str]]:
    encoded = tokenizer(
        chunk,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )
    token_lengths = [len(ids) for ids in encoded["input_ids"]]
    chopped_char_lengths = [_compute_char_len_from_offsets(offsets) for offsets in encoded["offset_mapping"]]
    chopped_texts = [text[:cl] for text, cl in zip(chunk, chopped_char_lengths)]
    word_lengths = [len(t.split()) for t in chopped_texts]
    return token_lengths, word_lengths, chopped_char_lengths, chopped_texts


def _process_chunk_decode(
    chunk: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> tuple[list[int], list[int], list[int], list[str]]:
    encoded = tokenizer(
        chunk,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"]
    token_lengths = [len(ids) for ids in input_ids]
    decoded_texts = _batch_decode_cached_specials(input_ids, tokenizer)

    word_lengths: list[int] = []
    chopped_char_lengths: list[int] = []
    chopped_texts: list[str] = []

    append_word_length = word_lengths.append
    append_char_length = chopped_char_lengths.append
    append_text = chopped_texts.append

    for original, decoded in zip(chunk, decoded_texts):
        if len(decoded) < len(original):
            char_len = _align_decoded_to_original(original, decoded)
            text = original[:char_len]
        else:
            char_len = len(original)
            text = original

        append_word_length(len(text.split()))
        append_char_length(char_len)
        append_text(text)

    return token_lengths, word_lengths, chopped_char_lengths, chopped_texts


def compute_lengths(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    max_workers: int = 4,
) -> tuple[list[int], list[int], list[int], list[str]]:
    """
    Tokenize `texts` in parallel and return per-sequence length statistics.

    Returns a tuple of `(token_lengths, word_lengths, char_lengths, truncated_texts)`
    where each list is aligned with the input `texts`.  Sequences longer than
    `max_length` tokens are silently truncated and the shorter version is
    included in `truncated_texts`.
    """
    strategy = "hf_offsets" if _supports_offset_mapping(tokenizer) else "decode"
    worker_fn = _process_chunk_hf_offsets if strategy == "hf_offsets" else _process_chunk_decode

    def _batch_fn(batch: dict) -> dict:
        tl, wl, cl, tx = worker_fn(batch["texts"], tokenizer, max_length)
        return {"token_lengths": tl, "word_lengths": wl, "char_lengths": cl, "texts": tx}

    desc = f"Step 1: Get Lengths"
    if strategy != "hf_offsets":
        desc += "[Slow Mode: Tokenizer lacks hf_offsets]"

    dataset = HuggingFaceDataset.from_list([{"texts": t} for t in texts])
    dataset = dataset.map(
        _batch_fn,
        batched=True,
        batch_size=200,
        num_proc=max_workers,
        remove_columns=["texts"],
        desc=desc,
    )

    return (
        dataset["token_lengths"],
        dataset["word_lengths"],
        dataset["char_lengths"],
        dataset["texts"],
    )


def _effective_num_workers(num_workers: int, debug: bool) -> int:
    if debug:
        return 0
    available_workers = os.cpu_count() or 1
    return min(available_workers, num_workers)


def _validate_precomputed_lengths(
    texts: list[str],
    token_lengths: list[int] | None,
    word_lengths: list[int] | None,
    char_lengths: list[int] | None,
) -> tuple[list[int], list[int], list[int]] | None:
    provided_lengths = [token_lengths, word_lengths, char_lengths]
    if all(lengths is None for lengths in provided_lengths):
        return None

    if any(lengths is None for lengths in provided_lengths):
        raise ValueError(
            "When providing precomputed lengths, you must pass all of "
            "`token_lengths`, `word_lengths`, and `char_lengths`."
        )

    text_count = len(texts)
    if len(token_lengths) != text_count or len(word_lengths) != text_count or len(char_lengths) != text_count:
        raise ValueError(
            "Precomputed lengths must match `len(texts)` for token_lengths, " "word_lengths, and char_lengths."
        )

    return token_lengths, word_lengths, char_lengths


def _build_dynabatch_sampler_and_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_input_token_length: int,
    threshold: float,
    max_batch_range: float,
    shuffle: bool,
    shuffle_seed: int,
    shuffle_keep_first_n: int,
    friendly_batch_size: bool,
    keep_batch_size_even: bool,
    num_workers: int,
    debug: bool,
    dynamic_batch_mode: bool,
    smooth_batches: bool,
    smooth_batches_max_diff: float,
    save_input_features: bool = False,
    token_lengths: list[int] | None = None,
    word_lengths: list[int] | None = None,
    char_lengths: list[int] | None = None,
) -> tuple[DynaBatchSampler, list[str], int]:
    num_workers = _effective_num_workers(num_workers, debug)
    precomputed_lengths = _validate_precomputed_lengths(
        texts=texts,
        token_lengths=token_lengths,
        word_lengths=word_lengths,
        char_lengths=char_lengths,
    )
    if precomputed_lengths is None:
        token_lengths, word_lengths, char_lengths, truncated_texts = compute_lengths(
            texts, tokenizer, max_input_token_length, max_workers=num_workers
        )
    else:
        token_lengths, word_lengths, char_lengths = precomputed_lengths
        truncated_texts = texts

    sampler = DynaBatchSampler(
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
        smooth_batches=smooth_batches,
        smooth_batches_max_diff=smooth_batches_max_diff,
        keep_batch_size_even=keep_batch_size_even,
        debug=debug,
        save_input_features=save_input_features,
    )
    return sampler, truncated_texts, num_workers


def dynabatch_sampler(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_input_token_length: int = 512,
    threshold: float = 0.65,
    max_batch_range: float = 2.0,
    shuffle: bool = False,
    shuffle_seed: int = 21,
    shuffle_keep_first_n: int = 3,
    friendly_batch_size: bool = False,
    keep_batch_size_even: bool = True,
    num_workers: int = 4,
    debug: bool = False,
    dynamic_batch_mode: bool = True,
    smooth_batches: bool = True,
    smooth_batches_max_diff: float = 0.2,
    save_input_features: bool = False,
    token_lengths: list[int] | None = None,
    word_lengths: list[int] | None = None,
    char_lengths: list[int] | None = None,
) -> DynaBatchSampler:
    """
    Build a length-sorted batch sampler with optional regressor-guided sizing.

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
           whose prediction is ≤ ``threshold``. The default ``threshold`` of ``0.65``
           leaves a margin below the ``1.0`` reference point.

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
                                kept; the largest kept size is used. Lower = more
                                conservative; higher = larger batches, more risk.
        max_batch_range:        Maximum batch-size multiplier relative to ``batch_size``.
        shuffle:                If True, shuffle the order of the pre-built batches.
                                Sequences within each batch remain length-similar.
        shuffle_seed:           Seed for the RNG used to shuffle batches.
        shuffle_keep_first_n:   Number of leading batches kept in original order even
                                when shuffling, to ensure the first few batches always
                                fit within the established memory envelope.
        friendly_batch_size:    If True, round batch sizes to the nearest power of 2
                                (or 3 x power of 2). Recommended for training workloads.
        keep_batch_size_even:   If True, round batch sizes to the nearest even number.
        num_workers:            Number of parallel workers for the length pre-pass
                                (not a PyTorch DataLoader; see ``build_dynabatch_dataloader``).
        debug:                  If True, disable parallel workers and print per-batch
                                sizing decisions. Use only during development.
        dynamic_batch_mode:     If False, all batches use exactly ``batch_size`` items
                                (equivalent to a standard MaxToken sampler).
        smooth_batches:         If True, apply a post-pass that smooths adjacent batch
                                sizes to avoid abrupt size jumps.
        smooth_batches_max_diff: Maximum allowed per-step growth between adjacent
                                batches, expressed as a fraction of ``batch_size``.
                                For example, ``0.2`` allows at most ``0.2 * batch_size``
                                additional items per step (still capped by
                                ``max_batch_range``/sampler max size).
        save_input_features:    If True, save the input features to a file. (Purely for debugging purposes)
        token_lengths:          Optional precomputed token lengths aligned with
                                ``texts``. When provided together with
                                ``word_lengths`` and ``char_lengths``,
                                ``compute_lengths`` is skipped.
        word_lengths:           Optional precomputed word lengths aligned with
                                ``texts``.
        char_lengths:           Optional precomputed character lengths aligned
                                with ``texts``.

    Returns:
        A ``DynaBatchSampler`` instance.
    """
    sampler, _, _ = _build_dynabatch_sampler_and_texts(
        texts,
        tokenizer,
        batch_size,
        max_input_token_length,
        threshold,
        max_batch_range,
        shuffle,
        shuffle_seed,
        shuffle_keep_first_n,
        friendly_batch_size,
        keep_batch_size_even,
        num_workers,
        debug,
        dynamic_batch_mode,
        smooth_batches,
        smooth_batches_max_diff,
        save_input_features,
        token_lengths,
        word_lengths,
        char_lengths,
    )
    return sampler


def build_dynabatch_dataloader(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_input_token_length: int = 512,
    threshold: float = 0.65,
    max_batch_range: float = 2.0,
    shuffle: bool = False,
    shuffle_seed: int = 21,
    shuffle_keep_first_n: int = 3,
    friendly_batch_size: bool = False,
    keep_batch_size_even: bool = True,
    num_workers: int = 4,
    debug: bool = False,
    dynamic_batch_mode: bool = True,
    smooth_batches: bool = True,
    smooth_batches_max_diff: float = 0.2,
    save_input_features: bool = False,
    token_lengths: list[int] | None = None,
    word_lengths: list[int] | None = None,
    char_lengths: list[int] | None = None,
    **tokenizer_kwargs: Any,
) -> DataLoader:
    """
    Drop-in ``DataLoader`` that uses the same dynamic batching as ``dynabatch_sampler``.

    Sampler arguments are documented on ``dynabatch_sampler``. Remaining keyword
    arguments are passed through to the tokenizer inside ``collate_fn`` (for example
    ``truncation=True``, ``max_length=…``) in addition to ``padding=True`` and
    ``return_tensors="pt"``.
    """
    sampler, truncated_texts, num_workers = _build_dynabatch_sampler_and_texts(
        texts,
        tokenizer,
        batch_size,
        max_input_token_length,
        threshold,
        max_batch_range,
        shuffle,
        shuffle_seed,
        shuffle_keep_first_n,
        friendly_batch_size,
        keep_batch_size_even,
        num_workers,
        debug,
        dynamic_batch_mode,
        smooth_batches,
        smooth_batches_max_diff,
        save_input_features,
        token_lengths,
        word_lengths,
        char_lengths,
    )

    dataset = TextDataset(truncated_texts)

    collate = partial(_collate_fn, tokenizer=tokenizer, **tokenizer_kwargs)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=num_workers)
