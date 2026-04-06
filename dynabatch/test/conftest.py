from pathlib import Path

import pytest
import torch
import random


_DATA_DIR = Path(__file__).parent / "data"
random.seed(21)

@pytest.fixture(scope="session")
def sample_texts(sample_size: int = 1000):
    path = _DATA_DIR / "human_written_data_ru_en.en"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]
    random.shuffle(lines)
    return lines[:sample_size]


class MockTokenizer:
    """Lightweight HuggingFace-compatible tokenizer backed by whitespace splitting."""

    def __call__(
        self,
        texts,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]

        tokenized = [text.split() for text in texts]

        if truncation and max_length is not None:
            tokenized = [t[:max_length] for t in tokenized]

        # Map each word to a stable integer ID (1-1000, 0 reserved for padding)
        token_ids = [[hash(w) % 1000 + 1 for w in tokens] for tokens in tokenized]

        if padding:
            max_len = max((len(ids) for ids in token_ids), default=0)
            attention_masks = [
                [1] * len(ids) + [0] * (max_len - len(ids)) for ids in token_ids
            ]
            token_ids = [
                ids + [0] * (max_len - len(ids)) for ids in token_ids
            ]
        else:
            attention_masks = [[1] * len(ids) for ids in token_ids]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

        return {"input_ids": token_ids, "attention_mask": attention_masks}


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()
