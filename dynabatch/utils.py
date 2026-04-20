import gc
import math
import traceback
from typing import Any

import torch
from transformers import TrainerCallback


def clear_gpu_memory(oom_error: Exception, **tensors: torch.Tensor) -> None:
    """
    Recover GPU memory after an OOM error.

    Clears the traceback frames to release references held by the exception,
    deletes the provided tensors, and runs the Python GC followed by
    ``torch.cuda.empty_cache()``.

    Args:
        oom_error: The caught OOM exception whose traceback frames will be freed.
        **tensors: Named tensors (or other objects) to delete before cache clearing.
    """
    traceback.clear_frames(oom_error.__traceback__)
    oom_error = None  # noqa: F841 — drop reference so the traceback can be freed
    for key in list(tensors.keys()):
        del tensors[key]
    gc.collect()
    torch.cuda.empty_cache()


def split_batch(batch: dict, chunk_size: int) -> list[dict]:
    """
    Splits a BatchEncoding (or any dict of tensors) into sub-batches, where each
    sub-batch contains at most ``chunk_size`` items.

    Args:
        batch:      A dict mapping string keys to tensors (or sequences) indexed
                    along the first dimension.
        chunk_size: Maximum number of items per sub-batch.

    Returns:
        A list of dicts with the same keys as ``batch``, each covering a
        contiguous slice of up to ``chunk_size`` items.
    """
    n_samples = len(batch["input_ids"])
    chunks = []
    for start in range(0, n_samples, chunk_size):
        chunk = {key: value[start : start + chunk_size] for key, value in batch.items()}
        chunks.append(chunk)
    return chunks


def split_inputs_dict(
    batch: dict[str, Any], chunk_size: int, batch_size_key: str = "input_ids"
) -> list[dict[str, Any]]:
    """
    Split trainer inputs into size-limited chunks while preserving metadata keys.

    Tensor values and sequence-like values that match the batch length are sliced
    on the first dimension. Scalar/config-like values are copied as-is.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")

    batch_size_value = batch.get(batch_size_key)
    if batch_size_value is None:
        raise KeyError(f"Batch size key `{batch_size_key}` not found in inputs.")

    n_samples = len(batch_size_value)
    chunks: list[dict[str, Any]] = []
    for start in range(0, n_samples, chunk_size):
        end = start + chunk_size
        chunk: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                chunk[key] = value[start:end]
            elif isinstance(value, (list, tuple)) and len(value) == n_samples:
                chunk[key] = value[start:end]
            else:
                chunk[key] = value
        chunks.append(chunk)
    return chunks


def merge_outputs(outputs: list[torch.Tensor]) -> torch.Tensor | None:
    """
    Merges a list of tensors from ``model.generate()`` into a single tensor.
    Handles variable sequence lengths by right-padding shorter tensors with zeros.

    Args:
        outputs: List of 2-D tensors with shape ``(batch_size, seq_len)``.

    Returns:
        A single concatenated tensor, or ``None`` if ``outputs`` is empty.
    """
    if not outputs:
        return None

    seq_lengths = [o.shape[1] for o in outputs]
    uniform_lengths = len(set(seq_lengths)) == 1

    if uniform_lengths:
        return torch.cat(outputs, dim=0)

    max_len = max(seq_lengths)
    padded_outputs = []
    for o in outputs:
        if o.shape[1] < max_len:
            pad_amount = max_len - o.shape[1]
            o = torch.nn.functional.pad(o, (0, pad_amount), value=0)
        padded_outputs.append(o)

    return torch.cat(padded_outputs, dim=0)


def generate_with_oom_fallback(
    model: Any,
    batch: dict,
    min_batch_size: int,
    device: torch.device,
    **generate_kwargs: Any,
) -> tuple[torch.Tensor, bool]:
    """
    Run ``model.generate()`` with OOM-safe fallback splitting.

    Attempts to generate from the full ``batch`` in one call. If a
    ``torch.cuda.OutOfMemoryError`` is raised, GPU memory is cleared, the
    batch is split into sub-batches of at most ``min_batch_size`` items, each
    sub-batch is generated independently with proper GPU cleanup between steps,
    and the results are merged back into a single tensor.

    Each sub-batch's output is immediately moved to CPU after generation so
    that only one sub-batch worth of activations lives on GPU at a time during
    recovery, minimising peak memory pressure.

    Args:
        model:            The generative model (must expose a ``generate`` method
                          compatible with HuggingFace's interface).
        batch:            A dict with at least ``"input_ids"`` and
                          ``"attention_mask"`` keys, whose values are CPU tensors
                          indexed along the first dimension.
        min_batch_size:   Maximum number of items per sub-batch during fallback
                          recovery. Use the same value passed as ``batch_size``
                          to ``build_dynamic_batch_dataloader``.
        device:           The torch device to move tensors onto before calling
                          ``generate``.
        **generate_kwargs: Additional keyword arguments forwarded to
                          ``model.generate()`` on every call (e.g.
                          ``forced_bos_token_id``).

    Returns:
        A tuple ``(output, did_fallback)`` where ``output`` is the merged
        generated token tensor and ``did_fallback`` is ``True`` if an OOM
        occurred and the fallback path was used.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    try:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        del input_ids, attention_mask
        return output, False
    except torch.cuda.OutOfMemoryError as oom_error:
        clear_gpu_memory(oom_error, input_ids=input_ids, attention_mask=attention_mask)
        sub_outputs = []
        for sub_batch in split_batch(batch, chunk_size=min_batch_size):
            sub_input = sub_batch["input_ids"].to(device)
            sub_mask = sub_batch["attention_mask"].to(device)
            sub_output = model.generate(
                input_ids=sub_input,
                attention_mask=sub_mask,
                **generate_kwargs,
            )
            sub_outputs.append(sub_output.cpu())
            del sub_input, sub_mask, sub_output
            torch.cuda.empty_cache()
        return merge_outputs(sub_outputs), True


def get_hardware_friendly_batch_size(target_size: int) -> int:
    """
    Returns the largest number <= target_size that is either
    a power of 2 (2^n) or 3 times a power of 2 (3 * 2^m).
    """
    if target_size < 1:
        raise ValueError("Batch size must be at least 1.")

    max_power_of_2 = 2 ** int(math.log2(target_size))
    max_3_times_power_of_2 = 0
    if target_size >= 3:
        max_3_times_power_of_2 = 3 * (2 ** int(math.log2(target_size / 3)))

    return max(max_power_of_2, max_3_times_power_of_2)


def get_even_batch_size(target_size: int) -> int:
    """
    Returns the largest number <= target_size that is an even number.
    """
    if target_size < 1:
        raise ValueError("Batch size must be at least 1.")
    elif target_size == 1:
        return 1

    return target_size if target_size % 2 == 0 else target_size - 1


class MemoryCleanupCallback(TrainerCallback):
    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def on_evaluate(self, args, state, control, **kwargs):
        """Runs immediately after the evaluation loop finishes."""
        self._cleanup()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Runs after the epoch is fully complete (including eval)."""
        self._cleanup()
