"""Internal module: length-sorted batch sampler with optional regressor-guided sizing."""

import random
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler
from tqdm import tqdm

from dynabatch.regressor import build_baseline_features, select_optimal_batch_size
from dynabatch.utils import get_hardware_friendly_batch_size

# Number of candidate batch size steps generated per unit of batch range.
# E.g. range (1.0 → 2.0) produces 20 candidates before deduplication.
_CANDIDATE_STEPS_PER_UNIT = 20


class MaxTokenBatchSampler(Sampler[list[int]]):
    """
    A batch sampler that sorts sequences by token length and optionally uses
    a pre-trained regressor to increase batch sizes for shorter sequences,
    keeping memory pressure close to the first (hardest) batch.

    Args:
        token_lengths:        Token count per sequence (pre-truncated).
        word_lengths:         Word count per sequence (pre-truncated).
        char_lengths:         Character count per sequence (pre-truncated).
        min_batch_size:       Batch size for the first (longest) batch; the
                              memory baseline.  Later batches may be larger.
        shuffle:              Shuffle the order of pre-built batches.
        threshold:            Maximum regressor prediction allowed for a
                              candidate batch.  See ``build_dynamic_batch_dataloader``
                              for the scale interpretation.
        max_batch_range:      Upper multiplier limit for candidate batch sizes
                              relative to ``min_batch_size``.
        shuffle_seed:         RNG seed used for batch-order shuffling.
        shuffle_keep_first_n: Number of leading batches kept in original order
                              even when shuffling.
        friendly_batch_size:  Round batch sizes to powers of 2 (or 3x powers of 2).
        dynamic_batch_mode:   When False, all batches use exactly ``min_batch_size``
                              items (equivalent to MaxTokenSampler).
        debug:                Print per-batch sizing decisions to stdout.
    """

    def __init__(
        self,
        token_lengths: list[int],
        word_lengths: list[int],
        char_lengths: list[int],
        min_batch_size: int,
        shuffle: bool = False,
        threshold: float = 0.7,
        max_batch_range: float = 2.0,
        shuffle_seed: int = 21,
        shuffle_keep_first_n: int = 5,
        friendly_batch_size: bool = False,
        dynamic_batch_mode: bool = True,
        debug: bool = False,
    ):
        sorted_indices = sorted(
            range(len(token_lengths)),
            key=lambda i: token_lengths[i],
            reverse=True,
        )
        self.min_batch_size = min_batch_size
        self.threshold = threshold
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_keep_first_n = shuffle_keep_first_n
        self.friendly_batch_size = friendly_batch_size
        self.dynamic_batch_mode = dynamic_batch_mode
        self.debug = debug

        self.batch_start_range = 1.0
        self.batch_end_range = max(max_batch_range, 1.0)
        self._n_steps = int((self.batch_end_range - self.batch_start_range) * _CANDIDATE_STEPS_PER_UNIT)

        self.batches = self._build_batches(sorted_indices, token_lengths, word_lengths, char_lengths)

    def _build_dynamic_batches(
        self,
        sorted_indices: list[int],
        token_lengths: list[int],
        word_lengths: list[int],
        char_lengths: list[int],
    ) -> list[list[int]]:
        sorted_token_lengths = np.array(token_lengths)[sorted_indices]
        sorted_word_lengths = np.array(word_lengths)[sorted_indices]
        sorted_char_lengths = np.array(char_lengths)[sorted_indices]

        multipliers = np.linspace(self.batch_start_range, self.batch_end_range, self._n_steps)
        candidate_batch_sizes = np.unique(np.round(self.min_batch_size * multipliers).astype(int))

        # The first batch always contains the longest sequences and uses the
        # minimum batch size — this is the hardest batch and the baseline for
        # all subsequent regressor predictions.
        batches: list[list[int]] = [sorted_indices[: self.min_batch_size]]

        baseline_features = build_baseline_features(
            token_lengths=sorted_token_lengths,
            word_lengths=sorted_word_lengths,
            char_lengths=sorted_char_lengths,
            min_batch_size=self.min_batch_size,
            n_candidates=len(candidate_batch_sizes),
        )

        remaining_token = sorted_token_lengths[self.min_batch_size :]
        remaining_word = sorted_word_lengths[self.min_batch_size :]
        remaining_char = sorted_char_lengths[self.min_batch_size :]
        next_start_idx = self.min_batch_size

        with tqdm(total=len(remaining_token), desc="Step 2: building dynamic batches", unit="seq") as pbar:
            while len(remaining_token):
                optimal_size = select_optimal_batch_size(
                    token_lengths=remaining_token,
                    word_lengths=remaining_word,
                    char_lengths=remaining_char,
                    baseline_features=baseline_features,
                    threshold=self.threshold,
                    candidate_batch_sizes=candidate_batch_sizes,
                )
                if self.friendly_batch_size:
                    optimal_size = get_hardware_friendly_batch_size(optimal_size)

                if len(remaining_token) <= optimal_size:
                    optimal_size = len(remaining_token)

                batch_indices = [sorted_indices[next_start_idx + i] for i in range(optimal_size)]
                if self.shuffle:
                    random.shuffle(batch_indices)
                batches.append(batch_indices)

                remaining_token = remaining_token[optimal_size:]
                remaining_word = remaining_word[optimal_size:]
                remaining_char = remaining_char[optimal_size:]
                next_start_idx += optimal_size
                pbar.update(optimal_size)

                if self.debug:
                    print(f"Batch N: {len(batches)} \t|\t Batch Size: {optimal_size}")
                    print("-" * 40)

        return batches

    def _build_static_batches(self, sorted_indices: list[int]) -> list[list[int]]:
        """Fixed size batching, equivalent to MaxTokenSampler."""
        batches = []
        for i in range(0, len(sorted_indices), self.min_batch_size):
            batch_indices = sorted_indices[i : i + self.min_batch_size]
            if self.shuffle:
                random.shuffle(batch_indices)
            batches.append(batch_indices)
        return batches

    def _build_batches(
        self,
        sorted_indices: list[int],
        token_lengths: list[int],
        word_lengths: list[int],
        char_lengths: list[int],
    ) -> list[list[int]]:
        random.seed(self.shuffle_seed)
        if self.dynamic_batch_mode:
            batches = self._build_dynamic_batches(sorted_indices, token_lengths, word_lengths, char_lengths)
        else:
            batches = self._build_static_batches(sorted_indices)

        if self.shuffle and len(batches) > self.shuffle_keep_first_n:
            first_n = batches[: self.shuffle_keep_first_n]
            rest = batches[self.shuffle_keep_first_n :]
            random.shuffle(rest)
            batches = first_n + rest

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)
