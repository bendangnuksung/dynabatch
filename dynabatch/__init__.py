import os

__version__ = "0.2.16"


def set_cuda_alloc_conf():
    """
    PyTorch's CUDA allocator normally reserves fixed-size memory blocks. When a block is too small
    for a new allocation, it has to find or create a new contiguous block. With
    expandable_segments:True, instead of reserving fixed blocks, the allocator can expand an
    existing segment using cuMemAddressReserve (the newer CUDA VMM API). This means:
        - Free blocks that are adjacent in virtual address space can be merged on demand
        - Large allocations no longer require a single pre-existing contiguous physical block
        - The allocator avoids the "memory is free but fragmented" failure mode
    This is useful when dynabatch SHUFFLE is set True because of the shuffle operations,
    the CUDA memory fragmentation gets worse and leading to OOM errors

    Important Note: this is only useful if it is set before CUDA / the CUDA allocator is initialized. So this
    package needs to be imported before  loading any Model to cuda or using any other CUDA related operations.
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


set_cuda_alloc_conf()

from .main import build_dynabatch_dataloader, compute_lengths, dynabatch_sampler
from .utils import MemoryCleanupCallback, clear_gpu_memory, generate_with_oom_fallback, merge_outputs, split_batch
