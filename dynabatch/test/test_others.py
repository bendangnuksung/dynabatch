import os

from dynabatch import set_cuda_alloc_conf
from dynabatch.sampler import DynaBatchSampler


def test_shuffle_batches_preserves_items():
    sampler = DynaBatchSampler(
        token_lengths=[9, 8, 7, 6, 5, 4, 3, 2],
        word_lengths=[9, 8, 7, 6, 5, 4, 3, 2],
        char_lengths=[90, 80, 70, 60, 50, 40, 30, 20],
        min_batch_size=2,
        shuffle=True,
        shuffle_seed=7,
        shuffle_keep_first_n=0,
        dynamic_batch_mode=False,
    )

    baseline_batches = [batch[:] for batch in sampler.batches]
    baseline_flat = sorted(idx for batch in baseline_batches for idx in batch)

    sampler._shufffle_batches()
    shuffled_batches = sampler.batches
    shuffled_flat = sorted(idx for batch in shuffled_batches for idx in batch)

    assert shuffled_flat == baseline_flat
    assert shuffled_batches != baseline_batches


def test_set_cuda_alloc_conf_preserves_existing_value(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    set_cuda_alloc_conf()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:256"
