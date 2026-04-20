"""
Trainer helpers to plug DynaBatch into HuggingFace Trainer classes.

Why this module exists:
- DynaBatch changes micro-batch sizes across steps.
- HuggingFace Trainer's default gradient-accumulation behavior assumes a stable
  per-step micro-batch size.
- For most real training runs this is still fine, but for strict baseline-vs-
  dynabatch comparisons, we often want explicit control over both:
  1) per-step loss contribution under variable micro-batch sizes, and
  2) learning-rate scaling when dynabatch changes steps-per-epoch.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from dynabatch.sampler import DynaBatchSampler
from dynabatch.utils import clear_gpu_memory, split_inputs_dict

_DYNABATCH_TRAINER_CACHE: dict[type, type] = {}
logger = logging.getLogger(__name__)

OOMFallbackStrategy = Literal["split_retry", "skip", None]

try:
    _TORCH_OOM_ERROR = torch.cuda.OutOfMemoryError
except AttributeError:

    class _TorchOOMError(RuntimeError):
        """Fallback OOM error type for very old torch versions."""

    _TORCH_OOM_ERROR = _TorchOOMError


def scale_lr_for_dynabatch(
    args: Any,
    sampler: DynaBatchSampler,
    dataset_size: int,
    baseline_batch_size: int | None = None,
) -> Any:
    """
    Return a copy of TrainingArguments with LR scaled for DynaBatch step count.

    The scaling multiplier is:

        static_steps_per_epoch / dynabatch_steps_per_epoch

    where ``static_steps_per_epoch`` is approximated with a fixed batch size and
    ``dynabatch_steps_per_epoch`` is ``len(sampler)``.

    Why this can matter:
    - DynaBatch can reduce the number of optimizer updates per epoch by packing
      more examples into later (shorter) batches.
    - With schedulers such as cosine decay, fewer updates compress the same LR
      schedule into fewer steps, which changes effective optimization dynamics.
    - Scaling LR by the step-count ratio is a practical linear-scaling-style
      correction for apples-to-apples comparisons against fixed-batch training.

    Why this is optional (and default-off in the mixin):
    - In normal production training, users often tune LR directly for their
      actual pipeline and do not want automatic LR transformation.
    - This helper is mainly for controlled comparisons with older trainer/
      collator setups that used fixed batch sizes.
    """
    if baseline_batch_size is None:
        baseline_batch_size = int(args.per_device_train_batch_size)

    if baseline_batch_size <= 0:
        raise ValueError("baseline_batch_size must be > 0.")

    static_steps_per_epoch = max(1, dataset_size // baseline_batch_size)
    dynabatch_steps_per_epoch = max(1, len(sampler))

    scaled_args = deepcopy(args)
    scaled_args.learning_rate = scaled_args.learning_rate * (static_steps_per_epoch / dynabatch_steps_per_epoch)
    return scaled_args


class DynabatchTrainerMixin:
    """
    Mixin that injects DynaBatch dataloader and variable-batch loss scaling.

    Design notes:
    - Uses ``super().compute_loss(...)`` to preserve trainer-specific behavior
      (label smoothing, subclass hooks, custom loss paths) and only applies
      dynabatch-specific rescaling as a post-process.
    - ``auto_scale_lr`` defaults to ``False`` because automatic LR scaling is
      usually needed for benchmark fairness, not for day-to-day training.
    """

    def __init__(
        self,
        *args: Any,
        dynabatch_sampler: DynaBatchSampler,
        auto_scale_lr: bool = False,
        batch_size_key: str = "input_ids",
        oom_fallback: OOMFallbackStrategy = None,
        oom_min_batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._dynabatch_sampler = dynabatch_sampler
        self._batch_size_key = batch_size_key
        self._oom_fallback = oom_fallback
        self._oom_min_batch_size = oom_min_batch_size
        self._oom_fallback_state: dict[str, int] | None = None
        self._oom_failed_count = 0

        if auto_scale_lr:
            # Auto LR scaling is intentionally opt-in. It is most useful when
            # reproducing old fixed-batch baselines and keeping comparable
            # optimization signal per epoch after dynabatch changes step count.
            training_args = kwargs.get("args")
            train_dataset = kwargs.get("train_dataset")
            if training_args is None or train_dataset is None:
                raise ValueError("auto_scale_lr=True requires `args` and `train_dataset` as keyword arguments.")
            kwargs["args"] = scale_lr_for_dynabatch(
                args=training_args,
                sampler=dynabatch_sampler,
                dataset_size=len(train_dataset),
            )

        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._dynabatch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_expected_batch_size(self) -> int:
        return int(self.args.per_device_train_batch_size)

    def _get_current_batch_size(self, inputs: dict[str, Any]) -> int:
        if self._batch_size_key not in inputs:
            raise KeyError(
                f"Batch size key `{self._batch_size_key}` not found in inputs. "
                "Pass `batch_size_key=...` for your modality."
            )
        return int(inputs[self._batch_size_key].shape[0])

    def _resolve_oom_min_batch_size(self) -> int:
        if self._oom_min_batch_size is not None:
            if self._oom_min_batch_size <= 0:
                raise ValueError("oom_min_batch_size must be > 0.")
            return int(self._oom_min_batch_size)

        sampler_min_size = int(getattr(self._dynabatch_sampler, "min_batch_size", 1))
        return max(1, sampler_min_size)

    def _split_inputs(self, inputs: dict[str, Any], chunk_size: int) -> list[dict[str, Any]]:
        return split_inputs_dict(
            batch=inputs,
            chunk_size=chunk_size,
            batch_size_key=self._batch_size_key,
        )

    def _recover_oom(self, oom_error: BaseException, **tensors: torch.Tensor) -> None:
        clear_gpu_memory(oom_error, **tensors)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _get_model_device(self, model: torch.nn.Module) -> torch.device:
        parameter = next(model.parameters(), None)
        if parameter is not None:
            return parameter.device
        return torch.device("cpu")

    def _record_oom_failure(self) -> None:
        self._oom_failed_count += 1
        if hasattr(self, "log"):
            self.log({"oom_failed": self._oom_failed_count})

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        current_batch_size = self._get_current_batch_size(inputs)
        loss, outputs = super().compute_loss(
            model=model,
            inputs=inputs,
            return_outputs=True,
            **kwargs,
        )

        if self._oom_fallback_state is not None:
            if self.args.gradient_accumulation_steps > 1:
                loss = loss * (current_batch_size / self._get_expected_batch_size())
            else:
                total_batch_size = max(1, int(self._oom_fallback_state["total_size"]))
                loss = loss * (current_batch_size / total_batch_size)
            return (loss, outputs) if return_outputs else loss

        if self.args.gradient_accumulation_steps > 1:
            # Why this reweighting is needed:
            # - Under gradient accumulation, Trainer aggregates micro-batch losses
            #   assuming a stable micro-batch size.
            # - DynaBatch deliberately varies micro-batch size by step.
            # - Without reweighting, smaller/larger micro-batches can contribute
            #   disproportionately to the accumulated optimizer update.
            # - Scaling by (current/expected batch size) makes each micro-batch
            #   contribute roughly in proportion to sample count, which is the
            #   intended behavior for variable-size accumulation windows.
            loss = loss * (current_batch_size / self._get_expected_batch_size())

        return (loss, outputs) if return_outputs else loss

    def _handle_oom_skip(self, model: torch.nn.Module, reason: str = "full-batch step OOM") -> torch.Tensor:
        logger.warning("Skipping training step after %s.", reason)
        return torch.zeros((), device=self._get_model_device(model))

    def _handle_oom_split_retry(self, model, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
        total_size = self._get_current_batch_size(inputs)
        chunk_size = self._resolve_oom_min_batch_size()
        chunks = self._split_inputs(inputs, chunk_size=chunk_size)

        logger.warning(
            "OOM fallback engaged: retrying training step in %d chunk(s) of <= %d samples.",
            len(chunks),
            chunk_size,
        )

        accumulated_loss: torch.Tensor | None = None
        successful_chunks = 0
        self._oom_fallback_state = {"total_size": total_size}
        try:
            for chunk in chunks:
                try:
                    chunk_loss = super().training_step(model, chunk, *args, **kwargs)
                    accumulated_loss = chunk_loss if accumulated_loss is None else accumulated_loss + chunk_loss
                    successful_chunks += 1
                except _TORCH_OOM_ERROR as sub_oom_error:
                    self._recover_oom(sub_oom_error)
                    self._record_oom_failure()
                    logger.warning(
                        "Sub-chunk OOM during split_retry fallback. Chunk size %d was skipped.",
                        self._get_current_batch_size(chunk),
                    )
        finally:
            self._oom_fallback_state = None

        if accumulated_loss is None:
            return self._handle_oom_skip(model, reason="OOM in all fallback chunks")

        if successful_chunks < len(chunks):
            logger.warning(
                "OOM fallback completed with partial chunks: %d/%d succeeded.",
                successful_chunks,
                len(chunks),
            )

        return accumulated_loss

    def training_step(self, model, inputs, *args: Any, **kwargs: Any):
        try:
            return super().training_step(model, inputs, *args, **kwargs)
        except _TORCH_OOM_ERROR as oom_error:
            self._recover_oom(oom_error)
            self._record_oom_failure()
            if self._oom_fallback is None:
                raise
            if self._oom_fallback == "skip":
                return self._handle_oom_skip(model)
            if self._oom_fallback == "split_retry":
                return self._handle_oom_split_retry(model, inputs, *args, **kwargs)
            raise ValueError("oom_fallback must be one of: 'split_retry', 'skip', or None.")


def make_dynabatch_trainer(trainer_cls: type) -> type:
    """Return (and cache) a Dynabatch-enabled subclass for a Trainer class."""
    cached = _DYNABATCH_TRAINER_CACHE.get(trainer_cls)
    if cached is not None:
        return cached

    class _DynabatchTrainer(DynabatchTrainerMixin, trainer_cls):
        pass

    _DynabatchTrainer.__name__ = f"Dynabatch{trainer_cls.__name__}"
    _DynabatchTrainer.__qualname__ = _DynabatchTrainer.__name__
    _DYNABATCH_TRAINER_CACHE[trainer_cls] = _DynabatchTrainer
    return _DynabatchTrainer


__all__ = [
    "DynabatchTrainerMixin",
    "OOMFallbackStrategy",
    "make_dynabatch_trainer",
    "scale_lr_for_dynabatch",
]
