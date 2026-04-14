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


def _compute_char_len_from_offsets(sequence_offsets: list[tuple[int, int]] | list[list[int]]) -> int:
    real_offsets = [offset for offset in sequence_offsets if tuple(offset) != (0, 0)]
    return max((int(offset[1]) for offset in real_offsets), default=0)


def _bytes_to_char_index(text: str, byte_index: int) -> int:
    return len(text.encode("utf-8")[:byte_index].decode("utf-8", errors="ignore"))


def _extract_single_sequence_offsets(encoded_offsets: Any) -> list[tuple[int, int]] | list[list[int]]:
    if encoded_offsets is None or len(encoded_offsets) == 0:
        return []
    first_item = encoded_offsets[0]
    if isinstance(first_item, (tuple, list)) and len(first_item) == 2 and not isinstance(first_item[0], (tuple, list)):
        return encoded_offsets
    return encoded_offsets[0]


def _get_sentencepiece_model(tokenizer: PreTrainedTokenizerBase) -> Any | None:
    sp_model = getattr(tokenizer, "sp_model", None)
    if sp_model is not None:
        return sp_model
    nested_tokenizer = getattr(tokenizer, "tokenizer", None)
    if nested_tokenizer is not None:
        return getattr(nested_tokenizer, "sp_model", None)
    return None


def _get_immutable_proto(sp_model: Any, text: str) -> Any:
    if hasattr(sp_model, "encode_as_immutable_proto"):
        return sp_model.encode_as_immutable_proto(text)
    if hasattr(sp_model, "EncodeAsImmutableProto"):
        return sp_model.EncodeAsImmutableProto(text)
    raise AttributeError("SentencePiece model does not support immutable proto offsets.")


def _select_length_strategy(texts: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int) -> str:
    probe_text = texts[0] if texts else ""

    try:
        probe = tokenizer(
            text=[probe_text],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_offsets_mapping=True,
        )
        if "offset_mapping" in probe:
            return "batched_hf_offsets"
    except Exception:
        pass

    sp_model = _get_sentencepiece_model(tokenizer)
    if sp_model is not None:
        try:
            _ = _get_immutable_proto(sp_model, probe_text)
            return "sentencepiece_offsets"
        except Exception:
            pass

    raise RuntimeError(
        "Unable to compute exact char_lengths/word_lengths: tokenizer does not support "
        "HuggingFace offset_mapping in batch or per-example mode, and no SentencePiece immutable-proto "
        "offset API was found."
    )


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

    strategy = _select_length_strategy(texts=texts, tokenizer=tokenizer, max_length=max_length)

    def _compute_lengths_for_batch_with_hf_offsets(data: list[str]):
        encoded = tokenizer(
            text=data["texts"], truncation=True, max_length=max_length, padding=True, return_offsets_mapping=True
        )
        token_lengths = [int(sum(mask)) for mask in encoded["attention_mask"]]
        chopped_char_lengths = []
        for sequence_offsets in encoded["offset_mapping"]:
            chopped_char_lengths.append(_compute_char_len_from_offsets(sequence_offsets))
        chopped_texts = [text[: chopped_char_lengths[i]] for i, text in enumerate(data["texts"])]
        chopped_word_lengths = [len(text.split()) for text in chopped_texts]
        return {
            "token_lengths": token_lengths,
            "word_lengths": chopped_word_lengths,
            "char_lengths": chopped_char_lengths,
            "texts": chopped_texts,
        }

    def _compute_lengths_for_batch_with_sentencepiece(data: list[str]):
        sp_model = _get_sentencepiece_model(tokenizer)
        if sp_model is None:
            raise RuntimeError("SentencePiece fallback selected but tokenizer has no accessible sentencepiece model.")

        encoded = tokenizer(
            text=data["texts"],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_attention_mask=True,
        )
        token_lengths = [int(sum(mask)) for mask in encoded["attention_mask"]]
        chopped_char_lengths = []
        for text in data["texts"]:
            encoded_single = tokenizer(
                text=text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
            )
            input_ids = encoded_single["input_ids"]
            if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]

            special_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
            kept_text_piece_count = int(sum(1 for value in special_mask if value == 0))

            immutable_proto = _get_immutable_proto(sp_model, text)
            pieces = list(immutable_proto.pieces)
            if kept_text_piece_count <= 0 or not pieces:
                chopped_char_lengths.append(0)
                continue

            # If all text pieces are kept, no truncation happened.
            # SentencePiece piece end offsets can still undershoot raw-text length
            # due to normalization/whitespace handling, so keep full text here.
            if kept_text_piece_count >= len(pieces):
                chopped_char_lengths.append(len(text))
                continue

            capped_piece_count = min(kept_text_piece_count, len(pieces))
            end_byte = int(pieces[capped_piece_count - 1].end)
            chopped_char_length = _bytes_to_char_index(text, end_byte)
            # Preserve trailing whitespace/newlines that immediately follow the
            # last kept token boundary.
            while chopped_char_length < len(text) and text[chopped_char_length].isspace():
                chopped_char_length += 1
            chopped_char_lengths.append(chopped_char_length)

        chopped_texts = [text[: chopped_char_lengths[i]] for i, text in enumerate(data["texts"])]
        chopped_word_lengths = [len(text.split()) for text in chopped_texts]

        return {
            "token_lengths": token_lengths,
            "word_lengths": chopped_word_lengths,
            "char_lengths": chopped_char_lengths,
            "texts": chopped_texts,
        }

    df = pd.DataFrame({"texts": texts})
    datasets = DatasetDict({"data": HuggingFaceDataset.from_pandas(df)})

    map_fn = _compute_lengths_for_batch_with_hf_offsets
    if strategy == "sentencepiece_offsets":
        map_fn = _compute_lengths_for_batch_with_sentencepiece

    datasets = datasets.map(
        map_fn,
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
    threshold: float = 0.65,
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
                                (or 3 x power of 2). Recommended for training workloads.
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
