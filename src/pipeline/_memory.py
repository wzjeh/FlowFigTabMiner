"""Stage-boundary memory release for the unified pipeline.

On Mac unified memory (MPS backend) a bare ``gc.collect()`` is not enough
to hand freed tensors back to the OS — the MPS allocator keeps them in
its cache.  ``release_memory`` runs GC *and* empties the MPS/CUDA cache,
so heavy vision models (Florence-2, YOLO, TATR) actually free their
memory at stage boundaries instead of accumulating across Steps 1→6.

Use at every stage boundary where a heavy model goes out of scope.
"""

import gc


def release_memory() -> None:
    """GC + empty the accelerator allocator cache (MPS or CUDA)."""
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # torch absent or backend probe failed — GC already ran, which is
        # the best we can do.
        pass
