"""Internal module: length-sorted batch sampler with optional regressor-guided sizing."""

import os
import random
from itertools import chain
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler
from tqdm import tqdm

from dynabatch.regressor import build_baseline_features, select_optimal_batch_size
from dynabatch.utils import get_even_batch_size, get_hardware_friendly_batch_size

# Number of candidate batch size steps generated per unit of batch range.
# E.g. range (1.0 → 2.0) produces 20 candidates before deduplication.
_CANDIDATE_STEPS_PER_UNIT = 20


class DynaBatchSampler(Sampler[list[int]]):
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
                              even when shuffling. (Needed for early OOM detection.)
        keep_batch_size_even: Round batch sizes to even numbers.
        friendly_batch_size:  Round batch sizes to powers of 2 (or 3x powers of 2).
        dynamic_batch_mode:   When False, all batches use exactly ``min_batch_size``
                              items (equivalent to MaxTokenSampler).
        smooth_batches:       If True, apply a post-pass that smooths adjacent batch
                              sizes to avoid large step changes.
        smooth_batches_max_diff: Maximum allowed per-step growth between adjacent
                              batches, expressed as a fraction of ``min_batch_size``.
                              For example, ``0.2`` allows at most ``0.2 * min_batch_size``
                              additional items per step (still capped by
                              ``max_batch_range``/sampler max size).
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
        keep_batch_size_even: bool = False,
        friendly_batch_size: bool = False,
        dynamic_batch_mode: bool = True,
        smooth_batches: bool = True,
        smooth_batches_max_diff: float = 0.2,
        debug: bool = False,
    ):
        self._rng = random.Random(shuffle_seed)
        sorted_indices = sorted(
            range(len(token_lengths)),
            key=lambda i: token_lengths[i],
            reverse=True,
        )
        self.threshold = threshold
        self.shuffle = shuffle
        self.shuffle_keep_first_n = shuffle_keep_first_n
        self.is_first_shuffle = True
        self.friendly_batch_size = friendly_batch_size
        self.dynamic_batch_mode = dynamic_batch_mode
        self.debug = debug
        self.smooth_batches = smooth_batches
        self.smooth_batches_max_diff = smooth_batches_max_diff
        self.keep_batch_size_even = keep_batch_size_even

        self.batch_start_range = 1.0
        self.batch_end_range = max(max_batch_range, 1.0)
        self.min_batch_size = min_batch_size
        self.max_batch_size = int(self.batch_end_range * self.min_batch_size)
        self._n_steps = max(
            1,
            int((self.batch_end_range - self.batch_start_range) * _CANDIDATE_STEPS_PER_UNIT),
        )

        self.batches = self._build_batches(sorted_indices, token_lengths, word_lengths, char_lengths)

    def _get_safe_smooth_batch_max_diff(self) -> int:
        minimum_diff = 1 / max(1.0, self.max_batch_size - self.min_batch_size)
        return max(minimum_diff, self.smooth_batches_max_diff)

    def _smooth_batches(self, batches: list[list[int]]) -> list[list[int]]:
        """
        Re-balance dynamic batch lengths to avoid large step changes.

        The max increase from one batch to the next is capped by
        ``min_batch_size * smooth_batches_max_diff`` (and by ``max_batch_size``).
        Any overflow that cannot fit under that cap is carried to later batches,
        so all original items are preserved.
        """
        if not self.smooth_batches:
            return batches

        all_items = list(chain.from_iterable(batches))
        original_lengths = [len(batch) for batch in batches]

        smooth_batches_max_diff = self._get_safe_smooth_batch_max_diff()
        max_growth_step = int(self.min_batch_size * smooth_batches_max_diff)

        smooth_lengths = []
        carry_over = 0
        last_length = None

        for length in original_lengths:
            target_length = length + carry_over

            if last_length is None:
                actual_length = min(target_length, self.max_batch_size)
            else:
                allowed_max = min(last_length + max_growth_step, self.max_batch_size)
                actual_length = min(target_length, allowed_max)

            smooth_lengths.append(actual_length)
            carry_over = target_length - actual_length
            last_length = actual_length

        while carry_over > 0:
            allowed_max = min(last_length + max_growth_step, self.max_batch_size)
            actual_length = min(carry_over, allowed_max)

            smooth_lengths.append(actual_length)
            carry_over -= actual_length
            last_length = actual_length

        smooth_batches_list = []
        current_idx = 0
        for length in smooth_lengths:
            smooth_batches_list.append(all_items[current_idx : current_idx + length])
            current_idx += length

        assert sum(smooth_lengths) == sum(original_lengths)

        if self.debug:
            print("\nSmoothening   \t  Before Batch Size\t| After Batch Size")
            for i in range(len(smooth_batches_list)):
                len_before = len(batches[i]) if i < len(batches) else 0
                print(f"Batch {i+1}:\t\t\t {len_before} \t|\t {len(smooth_batches_list[i])}")
                print("-" * 60)
        return smooth_batches_list

    def _arrange_batches(self, batches: list[list[int]], arrange_type: str) -> list[list[int]]:
        """
        Returns the batches arranged to have a friendly batch size.
        arrange_type can be "hardware_friendly" or "even".
        """
        if arrange_type == "hardware_friendly":
            if not self.friendly_batch_size:
                return batches
            func = get_hardware_friendly_batch_size
        elif arrange_type == "even":
            if not self.keep_batch_size_even:
                return batches
            func = get_even_batch_size
        else:
            raise ValueError(f"Invalid arrange type: {arrange_type}")

        all_items = list(chain.from_iterable(batches))
        arranged_batch_lengths = []
        original_lengths = [len(batch) for batch in batches]
        max_length = max(original_lengths)
        remaining_items = len(all_items)

        for i, length in enumerate(original_lengths):
            if i == len(original_lengths) - 1:
                length = min((length + remaining_items), max_length)
                length = length if length <= remaining_items else remaining_items
            arranged_length = func(length)
            arranged_batch_lengths.append(arranged_length)
            remaining_items -= arranged_length

        while remaining_items > 0:
            length = min(remaining_items, max_length)
            arranged_length = func(length)
            remaining_items -= arranged_length

            # If adding the remaining items to the arranged type length would not exceed the max length,
            # add them to the last batch
            if (arranged_length + remaining_items) <= max_length:
                arranged_length = arranged_length + remaining_items
                remaining_items = 0

            arranged_batch_lengths.append(arranged_length)

        arranged_batches = []
        current_idx = 0
        for length in arranged_batch_lengths:
            arranged_batches.append(all_items[current_idx : current_idx + length])
            current_idx += length

        assert sum(arranged_batch_lengths) == sum(original_lengths)

        if self.debug:
            print(f"\n{arrange_type.upper()} \t\t  Before Size\t| After Size")
            for i in range(len(arranged_batches)):
                len_before = len(batches[i]) if i < len(batches) else 0
                print(f"Batch {i+1}:\t\t\t {len_before} \t|\t {len(arranged_batches[i])}")
                print("-" * 60)

        return arranged_batches

    def _shufffle_batches(self):
        if not self.shuffle:
            return
        # shuffle indices within each batch
        [self._rng.shuffle(batch) for batch in self.batches]

        # shuffle batches
        if len(self.batches) > self.shuffle_keep_first_n and self.is_first_shuffle:
            first_n = self.batches[: self.shuffle_keep_first_n]
            rest = self.batches[self.shuffle_keep_first_n :]
            self._rng.shuffle(rest)
            self.batches = first_n + rest
            self.is_first_shuffle = False
        else:
            self._rng.shuffle(self.batches)

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

        with tqdm(total=len(sorted_token_lengths), desc="Step 2: building dynamic batches", unit="seq") as pbar:
            pbar.update(self.min_batch_size)
            while len(remaining_token):
                optimal_size = select_optimal_batch_size(
                    token_lengths=remaining_token,
                    word_lengths=remaining_word,
                    char_lengths=remaining_char,
                    baseline_features=baseline_features,
                    threshold=self.threshold,
                    candidate_batch_sizes=candidate_batch_sizes,
                )

                if len(remaining_token) <= optimal_size:
                    optimal_size = len(remaining_token)

                batch_indices = [sorted_indices[next_start_idx + i] for i in range(optimal_size)]
                batches.append(batch_indices)

                remaining_token = remaining_token[optimal_size:]
                remaining_word = remaining_word[optimal_size:]
                remaining_char = remaining_char[optimal_size:]
                next_start_idx += optimal_size
                pbar.update(optimal_size)

                if self.debug:
                    print(f"Batch {len(batches)} \t|\t Batch Size: {optimal_size}")
                    print("-" * 40)

        batches = self._smooth_batches(batches)
        batches = self._arrange_batches(batches, arrange_type="hardware_friendly")
        batches = self._arrange_batches(batches, arrange_type="even")
        return batches

    def _build_static_batches(self, sorted_indices: list[int]) -> list[list[int]]:
        """Fixed size batching, equivalent to MaxTokenSampler."""
        batches = []
        for i in range(0, len(sorted_indices), self.min_batch_size):
            batch_indices = sorted_indices[i : i + self.min_batch_size]
            batches.append(batch_indices)
        return batches

    def _build_batches(
        self,
        sorted_indices: list[int],
        token_lengths: list[int],
        word_lengths: list[int],
        char_lengths: list[int],
    ) -> list[list[int]]:
        if self.dynamic_batch_mode:
            batches = self._build_dynamic_batches(sorted_indices, token_lengths, word_lengths, char_lengths)
        else:
            batches = self._build_static_batches(sorted_indices)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        self._shufffle_batches()
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)
