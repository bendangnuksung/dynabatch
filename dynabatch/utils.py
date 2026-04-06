import gc
import math
import traceback

import torch


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
