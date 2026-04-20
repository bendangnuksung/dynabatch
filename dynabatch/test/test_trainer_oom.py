from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from dynabatch.trainer import make_dynabatch_trainer


class DummySampler:
    def __init__(self, min_batch_size: int = 2) -> None:
        self.min_batch_size = min_batch_size


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


class BaseTrainerStub:
    def __init__(
        self,
        *init_args,
        args=None,
        train_dataset=None,
        fail_once_on_batch_sizes=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.backward_log: list[float] = []
        self.logged_metrics: list[dict[str, float]] = []
        self._remaining_oom_by_size = dict(fail_once_on_batch_sizes or {})

    def log(self, metrics: dict[str, float]) -> None:
        self.logged_metrics.append(metrics)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = inputs["input_ids"].float().mean()
        return (loss, {"loss": loss}) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        batch_size = int(inputs["input_ids"].shape[0])
        if self._remaining_oom_by_size.get(batch_size, 0) > 0:
            self._remaining_oom_by_size[batch_size] -= 1
            raise torch.cuda.OutOfMemoryError(f"synthetic oom for batch_size={batch_size}")

        loss = self.compute_loss(model, inputs, return_outputs=False, **kwargs)
        detached = loss.detach()
        self.backward_log.append(float(detached))
        return detached


def _build_args(gradient_accumulation_steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )


def _build_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "attention_mask": torch.ones((4, 1)),
        "labels": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    }


def _build_trainer(
    *,
    gradient_accumulation_steps: int,
    oom_fallback: str | None = "split_retry",
    fail_once_on_batch_sizes: dict[int, int] | None = None,
    oom_min_batch_size: int | None = 2,
):
    trainer_cls = make_dynabatch_trainer(BaseTrainerStub)
    return trainer_cls(
        dynabatch_sampler=DummySampler(min_batch_size=2),
        args=_build_args(gradient_accumulation_steps),
        oom_fallback=oom_fallback,
        oom_min_batch_size=oom_min_batch_size,
        fail_once_on_batch_sizes=fail_once_on_batch_sizes or {},
    )


def test_split_retry_recovers_from_training_step_oom() -> None:
    trainer = _build_trainer(
        gradient_accumulation_steps=1,
        fail_once_on_batch_sizes={4: 1},
    )
    model = DummyModel()

    loss = trainer.training_step(model, _build_inputs())

    assert float(loss) == pytest.approx(2.5)
    assert trainer.backward_log == pytest.approx([0.75, 1.75])
    assert trainer._oom_failed_count == 1
    assert trainer.logged_metrics[-1] == {"oom_failed": 1}


def test_skip_strategy_returns_zero_and_continues() -> None:
    trainer = _build_trainer(
        gradient_accumulation_steps=1,
        oom_fallback="skip",
        fail_once_on_batch_sizes={4: 1},
    )
    model = DummyModel()

    loss = trainer.training_step(model, _build_inputs())

    assert float(loss) == 0.0
    assert trainer.backward_log == []
    assert trainer._oom_failed_count == 1
    assert trainer.logged_metrics[-1] == {"oom_failed": 1}


@pytest.mark.parametrize("gradient_accumulation_steps", [1, 2])
def test_split_retry_matches_full_batch_scaling(gradient_accumulation_steps: int) -> None:
    inputs = _build_inputs()
    model = DummyModel()

    baseline = _build_trainer(gradient_accumulation_steps=gradient_accumulation_steps)
    baseline_loss = baseline.training_step(model, inputs)

    fallback = _build_trainer(
        gradient_accumulation_steps=gradient_accumulation_steps,
        fail_once_on_batch_sizes={4: 1},
    )
    fallback_loss = fallback.training_step(model, inputs)

    assert float(fallback_loss) == pytest.approx(float(baseline_loss), rel=1e-6)
    assert sum(fallback.backward_log) == pytest.approx(sum(baseline.backward_log), rel=1e-6)


def test_split_retry_skips_subchunk_when_it_still_ooms(caplog: pytest.LogCaptureFixture) -> None:
    trainer = _build_trainer(
        gradient_accumulation_steps=1,
        fail_once_on_batch_sizes={4: 1, 2: 1},
    )
    model = DummyModel()

    with caplog.at_level(logging.WARNING):
        loss = trainer.training_step(model, _build_inputs())

    assert float(loss) == pytest.approx(1.75)
    assert trainer.backward_log == pytest.approx([1.75])
    assert trainer._oom_failed_count == 2
    assert trainer.logged_metrics[-1] == {"oom_failed": 2}
    assert "Sub-chunk OOM during split_retry fallback" in caplog.text
