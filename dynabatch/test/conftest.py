import os
import random
from pathlib import Path

# Prevent HuggingFace datasets from making any network requests during tests.
# Without this, importing `datasets` triggers HF Hub auth checks that add
# tens of seconds of latency even before a single test runs.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import datasets as _hf_datasets
import pytest
import torch

# Disable Arrow disk cache so any direct compute_lengths calls in tests don't
# write to ~/.cache/huggingface/datasets, which is the main source of latency.
_hf_datasets.disable_caching()

_DATA_DIR = Path(__file__).parent / "data"
random.seed(21)


@pytest.fixture(scope="session")
def sample_texts(sample_size: int = 200):
    path = _DATA_DIR / "human_written_data_ru_en.en"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]
    random.shuffle(lines)
    return lines[:sample_size]


class MockTokenizer:
    """Lightweight HuggingFace-compatible tokenizer backed by whitespace splitting."""

    def __call__(
        self,
        text: str | list[str] | None = None,
        text_pair=None,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_offsets_mapping=False,
        **kwargs,
    ):
        # HuggingFace uses ``text=``; accept legacy ``texts=`` for older call sites.
        if text is None and "texts" in kwargs:
            text = kwargs.pop("texts")
        if text is None:
            raise TypeError("MockTokenizer.__call__() missing required argument: 'text'")
        if isinstance(text, str):
            texts_list = [text]
        else:
            texts_list = list(text)

        tokenized = [t.split() for t in texts_list]

        if truncation and max_length is not None:
            tokenized = [t[:max_length] for t in tokenized]

        # Map each word to a stable integer ID (1-1000, 0 reserved for padding)
        token_ids = [[hash(w) % 1000 + 1 for w in tokens] for tokens in tokenized]

        if padding:
            max_len = max((len(ids) for ids in token_ids), default=0)
            attention_masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in token_ids]
            token_ids = [ids + [0] * (max_len - len(ids)) for ids in token_ids]
        else:
            max_len = None
            attention_masks = [[1] * len(ids) for ids in token_ids]

        result: dict = {"input_ids": token_ids, "attention_mask": attention_masks}

        if return_offsets_mapping:
            all_offsets = []
            for original_text, words in zip(texts_list, tokenized):
                offsets = []
                pos = 0
                for word in words:
                    idx = original_text.find(word, pos)
                    if idx != -1:
                        offsets.append((idx, idx + len(word)))
                        pos = idx + len(word)
                    else:
                        offsets.append((0, 0))
                if max_len is not None:
                    offsets += [(0, 0)] * (max_len - len(offsets))
                all_offsets.append(offsets)
            result["offset_mapping"] = all_offsets

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(result["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(result["attention_mask"], dtype=torch.long),
                **({} if not return_offsets_mapping else {"offset_mapping": result["offset_mapping"]}),
            }

        return result


@pytest.fixture(scope="session")
def mock_tokenizer_session():
    return MockTokenizer()


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture(scope="session", autouse=True)
def fast_classifier():
    """
    Replace the HistGradientBoostingClassifier loaded at module level with a
    trivial stub for the duration of the test session.

    The real classifier takes ~1.5 s per predict_proba call; with ~100 batches
    per 200-text sampler that becomes 150 s per test.  Tests exercise batching
    logic and pipeline correctness, not the classifier's numeric output, so a
    stub that always returns 'safe' (low spike probability) is sufficient.
    """
    import dynabatch.main as _main

    real_clf = _main._CLASSIFIER

    class _StubClassifier:
        feature_names_in_ = real_clf.feature_names_in_

        def predict_proba(self, df):
            import numpy as np

            n = len(df)
            return np.column_stack([np.zeros(n), np.zeros(n)])

    _main._CLASSIFIER = _StubClassifier()
    yield
    _main._CLASSIFIER = real_clf


@pytest.fixture(scope="session")
def precomputed_lengths(sample_texts, mock_tokenizer_session):
    """
    Compute text lengths directly via the mock tokenizer, bypassing the HF
    Dataset machinery in compute_lengths.  Using compute_lengths here would
    incur Arrow serialisation + disk-cache overhead on every session, making
    the suite unnecessarily slow.  The values produced are identical to what
    compute_lengths returns for the same inputs and max_length=64.
    """
    max_length = 64
    encoded = mock_tokenizer_session(
        text=sample_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_offsets_mapping=True,
    )
    token_lengths = [sum(mask) for mask in encoded["attention_mask"]]
    char_lengths = []
    truncated_texts = []
    for offsets, text in zip(encoded["offset_mapping"], sample_texts):
        real_offsets = [o for o in offsets if o != (0, 0)]
        chopped = max((o[1] for o in real_offsets), default=0)
        char_lengths.append(chopped)
        truncated_texts.append(text[:chopped] if chopped else text)
    word_lengths = [len(t.split()) for t in truncated_texts]
    return token_lengths, word_lengths, char_lengths, truncated_texts
