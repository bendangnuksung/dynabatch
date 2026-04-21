from unittest.mock import patch

import pytest
import torch

from dynabatch.utils import clear_gpu_memory, get_hardware_friendly_batch_size, merge_outputs, split_batch

# ---------------------------------------------------------------------------
# split_batch
# ---------------------------------------------------------------------------


def _make_batch(n_samples: int, seq_len: int) -> dict:
    return {
        "input_ids": torch.ones(n_samples, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(n_samples, seq_len, dtype=torch.long),
    }


def test_split_batch_even():
    batch = _make_batch(6, 10)
    chunks = split_batch(batch, chunk_size=2)
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk["input_ids"].shape == (2, 10)
        assert chunk["attention_mask"].shape == (2, 10)


def test_split_batch_uneven():
    batch = _make_batch(7, 10)
    chunks = split_batch(batch, chunk_size=3)
    assert len(chunks) == 3
    assert chunks[0]["input_ids"].shape[0] == 3
    assert chunks[1]["input_ids"].shape[0] == 3
    assert chunks[2]["input_ids"].shape[0] == 1


def test_split_batch_single_chunk():
    batch = _make_batch(4, 10)
    chunks = split_batch(batch, chunk_size=10)
    assert len(chunks) == 1
    assert chunks[0]["input_ids"].shape == (4, 10)


def test_split_batch_preserves_values():
    ids = torch.arange(12).reshape(4, 3)
    batch = {"input_ids": ids, "attention_mask": torch.ones(4, 3)}
    chunks = split_batch(batch, chunk_size=2)
    assert torch.equal(chunks[0]["input_ids"], ids[:2])
    assert torch.equal(chunks[1]["input_ids"], ids[2:])


# ---------------------------------------------------------------------------
# merge_outputs
# ---------------------------------------------------------------------------


def test_merge_outputs_empty():
    result = merge_outputs([], pad_token_id=9)
    assert result is None


def test_merge_outputs_uniform():
    outputs = [torch.ones(2, 5), torch.ones(3, 5)]
    merged = merge_outputs(outputs, pad_token_id=9)
    assert merged is not None
    assert merged.shape == (5, 5)


def test_merge_outputs_variable():
    outputs = [torch.ones(2, 8), torch.ones(3, 4)]
    merged = merge_outputs(outputs, pad_token_id=9)
    assert merged is not None
    # rows from second tensor should be pad-token padded to length 8
    assert merged.shape == (5, 8)
    # first two rows from first tensor — all ones
    assert torch.all(merged[:2] == 1.0)
    # last three rows from second tensor — first 4 cols ones, last 4 are pad_token_id
    assert torch.all(merged[2:, :4] == 1.0)
    assert torch.all(merged[2:, 4:] == 9.0)


def test_merge_outputs_single():
    t = torch.arange(6).reshape(2, 3)
    merged = merge_outputs([t], pad_token_id=9)
    assert merged is not None
    assert torch.equal(merged, t)


# ---------------------------------------------------------------------------
# clear_gpu_memory
# ---------------------------------------------------------------------------


def test_clear_gpu_memory_calls_empty_cache():
    fake_tensor = torch.zeros(1)
    fake_oom = RuntimeError("CUDA out of memory")
    fake_oom.__traceback__ = None

    with patch("torch.cuda.empty_cache") as mock_cache:
        clear_gpu_memory(fake_oom, my_tensor=fake_tensor)
        mock_cache.assert_called_once()


def test_clear_gpu_memory_no_tensors():
    fake_oom = RuntimeError("CUDA out of memory")
    fake_oom.__traceback__ = None

    with patch("torch.cuda.empty_cache") as mock_cache:
        clear_gpu_memory(fake_oom)
        mock_cache.assert_called_once()


# ---------------------------------------------------------------------------
# get_hardware_friendly_batch_size
# ---------------------------------------------------------------------------


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _is_docstring_friendly(n: int) -> bool:
    """Matches the docstring: 2^n or 3 * 2^m."""
    if n < 1:
        return False
    if _is_power_of_2(n):
        return True
    if n % 3 != 0:
        return False
    m = n // 3
    return _is_power_of_2(m)


def _max_friendly_at_most(target: int) -> int:
    return max(x for x in range(1, target + 1) if _is_docstring_friendly(x))


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 4),
        (6, 6),
        (7, 6),
        (8, 8),
        (9, 8),
        (12, 12),
        (13, 12),
        (24, 24),
        (25, 24),
        (48, 48),
        (100, 96),
    ],
)
def test_get_hardware_friendly_batch_size_examples(target, expected):
    assert get_hardware_friendly_batch_size(target) == expected


@pytest.mark.parametrize("target", range(1, 513))
def test_get_hardware_friendly_batch_size_matches_bruteforce_max(target):
    assert get_hardware_friendly_batch_size(target) == _max_friendly_at_most(target)


def test_get_hardware_friendly_batch_size_rejects_non_positive():
    with pytest.raises(ValueError, match="at least 1"):
        get_hardware_friendly_batch_size(0)
    with pytest.raises(ValueError, match="at least 1"):
        get_hardware_friendly_batch_size(-1)
