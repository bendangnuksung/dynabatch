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


@pytest.fixture(scope="session")
def sample_texts_5000(sample_size: int = 5000):
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

        # Used by ``batch_decode`` in the decode-based length path (same process as ``__call__``).
        self._last_batch_words = [list(row) for row in tokenized]

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

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ) -> str:
        return self.batch_decode(
            [token_ids],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )[0]

    def batch_decode(
        self,
        sequences: list[list[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ) -> list[str]:
        words_batch = getattr(self, "_last_batch_words", None)
        if words_batch is not None and len(words_batch) == len(sequences):
            return [" ".join(words) for words in words_batch]
        # Best-effort when ``__call__`` did not set the scratch buffer (e.g. tests that only decode).
        return ["" for _ in sequences]


class WordTokenizer(MockTokenizer):
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
        if text is None and "texts" in kwargs:
            text = kwargs.pop("texts")
        if text is None:
            raise TypeError("WordTokenizer.__call__() missing required argument: 'text'")
        texts_list = [text] if isinstance(text, str) else list(text)
        tokenized = [t.split() for t in texts_list]
        self._last_batch_words = [list(row) for row in tokenized]
        token_ids = [[hash(w) % 1000 + 1 for w in tokens] for tokens in tokenized]
        r = {
            "input_ids": token_ids,
            "attention_mask": [[1] * len(ids) for ids in token_ids],
        }

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
                if max_length is not None:
                    offsets += [(0, 0)] * (max_length - len(offsets))
                all_offsets.append(offsets)
            r["offset_mapping"] = all_offsets
        return r


@pytest.fixture(scope="session")
def mock_tokenizer_session():
    return MockTokenizer()


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture(scope="session")
def mock_word_tokenizer_session():
    return WordTokenizer()


@pytest.fixture(scope="session", autouse=True)
def fast_regressor():
    """
    Replace the XGBRegressor with a trivial stub for the test session.

    The real model is fast enough per row, but many sampler batches multiply
    work; tests target batching and DataLoader behavior, not regressor scores.
    A stub that predicts 0.0 makes every candidate pass typical thresholds,
    matching a maximally permissive sizing choice.
    """
    import numpy as np

    import dynabatch.regressor as _reg_mod

    real_reg = _reg_mod.get_regressor()
    feature_names = list(real_reg.get_booster().feature_names)

    class _StubBooster:
        def __init__(self, names: list[str]):
            self.feature_names = names

    class _StubRegressor:
        def __init__(self, names: list[str]):
            self._booster = _StubBooster(names)

        def get_booster(self):
            return self._booster

        def predict(self, X):
            return np.zeros(len(X), dtype=np.float64)

    _reg_mod._regressor = _StubRegressor(feature_names)
    yield
    _reg_mod._regressor = real_reg


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
