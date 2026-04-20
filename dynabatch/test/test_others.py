from copy import deepcopy

from dynabatch.sampler import DynaBatchSampler

BATCHES_SAMPLES = [
    [1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11, 12],
    [13, 14],
]


def test_shuffle_batches():
    sum_batches = sorted([sum(batch) for batch in BATCHES_SAMPLES])

    DynaBatchSampler.shuffle = True
    DynaBatchSampler.batches = deepcopy(BATCHES_SAMPLES)
    DynaBatchSampler.is_first_shuffle = False
    DynaBatchSampler.shuffle_keep_first_n = 0
    DynaBatchSampler._shufffle_batches(DynaBatchSampler)
    dyna_batches = DynaBatchSampler.batches
    dyna_sum_batches = sorted([sum(batch) for batch in dyna_batches])

    assert BATCHES_SAMPLES != DynaBatchSampler.batches
    assert dyna_sum_batches == sum_batches
