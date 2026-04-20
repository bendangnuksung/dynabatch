from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

pytest.importorskip("transformers")

from dynabatch.trainer import DynabatchTrainerMixin, make_dynabatch_trainer, scale_lr_for_dynabatch


class _ListBatchSampler:
    def __init__(self, batches: list[list[int]]):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class _ToyDataset(Dataset):
    def __init__(self, size: int):
        self._data = [torch.tensor([i], dtype=torch.long) for i in range(size)]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return {"input_ids": self._data[idx]}


class _BaseTrainer:
    def __init__(self, *args, **kwargs):
        del args
        self.args = kwargs["args"]
        self.train_dataset = kwargs["train_dataset"]
        self.data_collator = kwargs.get("data_collator")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class _DynabatchTestTrainer(DynabatchTrainerMixin, _BaseTrainer):
    pass


class _ModelOutput:
    def __init__(self, loss: torch.Tensor):
        self.loss = loss


class _FixedLossModel:
    def __call__(self, **kwargs):
        del kwargs
        return _ModelOutput(loss=torch.tensor(2.0))


def _build_args(**overrides):
    defaults = dict(
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_scale_lr_for_dynabatch_returns_copy_and_expected_multiplier():
    args = _build_args(per_device_train_batch_size=8, learning_rate=2e-5)
    sampler = _ListBatchSampler([[0], [1], [2], [3], [4]])

    scaled = scale_lr_for_dynabatch(
        args=args,
        sampler=sampler,
        dataset_size=80,
    )

    assert scaled is not args
    assert args.learning_rate == 2e-5
    # static_steps = 80 // 8 = 10; dynabatch_steps = 5; multiplier = 2
    assert scaled.learning_rate == 4e-5


def test_dynabatch_compute_loss_scales_by_actual_batch_size():
    sampler = _ListBatchSampler([[0, 1], [2, 3]])
    trainer = _DynabatchTestTrainer(
        dynabatch_sampler=sampler,
        auto_scale_lr=False,
        args=_build_args(per_device_train_batch_size=8, gradient_accumulation_steps=4),
        train_dataset=_ToyDataset(size=4),
    )

    loss = trainer.compute_loss(
        model=_FixedLossModel(),
        inputs={"input_ids": torch.ones(4, 3, dtype=torch.long)},
        return_outputs=False,
    )

    # base loss 2.0 * (current_batch_size 4 / expected_batch_size 8) = 1.0
    assert float(loss) == 1.0


def test_dynabatch_get_train_dataloader_uses_given_sampler():
    sampler = _ListBatchSampler([[0, 1], [2, 3, 4]])
    trainer = _DynabatchTestTrainer(
        dynabatch_sampler=sampler,
        auto_scale_lr=False,
        args=_build_args(),
        train_dataset=_ToyDataset(size=5),
    )

    train_loader = trainer.get_train_dataloader()

    assert train_loader.batch_sampler is sampler


def test_make_dynabatch_trainer_caches_and_inherits():
    transformers = pytest.importorskip("transformers")
    Trainer = transformers.Trainer

    trainer_cls_1 = make_dynabatch_trainer(Trainer)
    trainer_cls_2 = make_dynabatch_trainer(Trainer)

    assert trainer_cls_1 is trainer_cls_2
    assert issubclass(trainer_cls_1, DynabatchTrainerMixin)
    assert issubclass(trainer_cls_1, Trainer)
