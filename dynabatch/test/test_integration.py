"""
Integration tests for build_dynamic_batch_dataloader.

Includes:
  - basic dataloader iteration / correctness
  - a mock training loop (forward + backward pass)
  - a mock inference loop (no_grad forward + merge_outputs)
"""

import math

import pytest
import torch
import torch.nn as nn

import dynabatch.main as _main_module
from dynabatch import build_dynabatch_dataloader
from dynabatch.utils import merge_outputs, split_batch

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BATCH_SIZE = 2
_MAX_TOKEN_LEN = 64
_VOCAB_SIZE = 1001  # must be > max token ID produced by MockTokenizer (max 1000)
_EMBED_DIM = 8
_OUTPUT_DIM = 4


class _SimpleModel(nn.Module):
    """Embed token IDs -> mean-pool -> linear projection."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(_VOCAB_SIZE, _EMBED_DIM, padding_idx=0)
        self.fc = nn.Linear(_EMBED_DIM, _OUTPUT_DIM)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T)
        embedded = self.embedding(input_ids)  # (B, T, embed_dim)
        pooled = embedded.mean(dim=1)  # (B, embed_dim)
        return self.fc(pooled)  # (B, output_dim)


def _build_loader(sample_texts, mock_tokenizer, shuffle=False):
    return build_dynabatch_dataloader(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        batch_size=_BATCH_SIZE,
        max_input_token_length=_MAX_TOKEN_LEN,
        shuffle=shuffle,
        num_workers=0,
        dynamic_batch_mode=True,
    )


def _build_static_loader(sample_texts, mock_tokenizer, shuffle=False):
    return build_dynabatch_dataloader(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        batch_size=_BATCH_SIZE,
        max_input_token_length=_MAX_TOKEN_LEN,
        shuffle=shuffle,
        num_workers=0,
        dynamic_batch_mode=False,
    )


@pytest.fixture(autouse=True)
def _patch_compute_lengths(monkeypatch, precomputed_lengths):
    """
    Replace compute_lengths inside build_dynamic_batch_dataloader with a fast
    stub that returns the session-cached result, eliminating repeated HF Dataset
    overhead on every integration test.
    """
    token_lengths, word_lengths, char_lengths, truncated_texts = precomputed_lengths

    def _fast_compute_lengths(texts, tokenizer, max_length, max_workers=4):
        return token_lengths, word_lengths, char_lengths, truncated_texts

    monkeypatch.setattr(_main_module, "compute_lengths", _fast_compute_lengths)


# ---------------------------------------------------------------------------
# test_build_dynamic_batch_dataloader
# ---------------------------------------------------------------------------


def test_dataloader_batch_keys(sample_texts, mock_tokenizer):
    loader = _build_loader(sample_texts, mock_tokenizer)
    for batch in loader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "texts" in batch


def test_dataloader_covers_all_texts(sample_texts, mock_tokenizer):
    from dynabatch.main import compute_lengths

    _, _, _, truncated_texts = compute_lengths(sample_texts, mock_tokenizer, max_length=_MAX_TOKEN_LEN)
    loader = _build_loader(sample_texts, mock_tokenizer)
    seen_texts = []
    for batch in loader:
        seen_texts.extend(batch["texts"])
    assert sorted(seen_texts) == sorted(truncated_texts)


def test_dataloader_no_duplicate_texts(sample_texts, mock_tokenizer):
    loader = _build_loader(sample_texts, mock_tokenizer)
    seen_texts = []
    for batch in loader:
        seen_texts.extend(batch["texts"])
    assert len(seen_texts) == len(sample_texts)


def test_dataloader_tensor_types(sample_texts, mock_tokenizer):
    loader = _build_loader(sample_texts, mock_tokenizer)
    for batch in loader:
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)
        assert batch["input_ids"].dtype == torch.long


def test_dataloader_first_batch_min_size(sample_texts, mock_tokenizer):
    loader = _build_loader(sample_texts, mock_tokenizer)
    first_batch = next(iter(loader))
    assert first_batch["input_ids"].shape[0] == _BATCH_SIZE


def test_dataloader_max_batch_range_kwarg(sample_texts, mock_tokenizer):
    """build_dynamic_batch_dataloader forwards max_batch_range to the sampler."""
    from dynabatch.main import compute_lengths

    _, _, _, truncated_texts = compute_lengths(sample_texts, mock_tokenizer, max_length=_MAX_TOKEN_LEN)
    loader = build_dynabatch_dataloader(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        batch_size=_BATCH_SIZE,
        max_input_token_length=_MAX_TOKEN_LEN,
        max_batch_range=2.5,
        shuffle=False,
        num_workers=0,
        dynamic_batch_mode=True,
    )
    seen = []
    for batch in loader:
        seen.extend(batch["texts"])
    assert sorted(seen) == sorted(truncated_texts)


# ---------------------------------------------------------------------------
# test_mock_training_loop
# ---------------------------------------------------------------------------


def test_mock_training_loop(sample_texts, mock_tokenizer):
    """
    Simulates a training loop:
      - iterate dataloader
      - run forward pass
      - compute dummy loss and backpropagate
      - verify gradients are populated
    """
    model = _SimpleModel()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loader = _build_loader(sample_texts, mock_tokenizer)

    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"]  # (B, T)

        # Optionally exercise split_batch: split into sub-batches of size 1
        sub_batches = split_batch(
            {"input_ids": input_ids, "attention_mask": batch["attention_mask"]},
            chunk_size=1,
        )

        for sub in sub_batches:
            optimizer.zero_grad()
            output = model(sub["input_ids"])  # (1, output_dim)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Gradients should have been computed for all parameters
    for param in model.parameters():
        assert param.grad is not None, f"Grad missing for {param.shape}"

    assert math.isfinite(total_loss), "Total loss should be finite"


# ---------------------------------------------------------------------------
# test_mock_inference_loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loader_builder", [_build_loader, _build_static_loader])
def test_mock_inference_loop(sample_texts, mock_tokenizer, loader_builder):
    """
    Simulates an inference loop:
      - iterate dataloader inside torch.no_grad()
      - collect per-batch outputs
      - merge into a single tensor with merge_outputs
      - verify the merged tensor covers all input texts
    """
    model = _SimpleModel()
    model.eval()

    loader = loader_builder(sample_texts, mock_tokenizer)

    outputs = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["input_ids"])  # (B, output_dim)
            outputs.append(out)

    assert len(outputs) > 0, "At least one batch must be produced"

    merged = merge_outputs(outputs)
    assert merged is not None
    # Each text produces exactly one output row
    assert merged.shape[0] == len(sample_texts)
    assert merged.shape[1] == _OUTPUT_DIM
