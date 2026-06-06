"""Process-wide thread caps — MUST be imported before torch / paddle / cv2.

On a 16-core machine each of PaddleOCR, PyTorch and OpenBLAS defaults to
spawning one worker thread per core.  With several model libraries live
at once that multiplies into 80+ thread contexts and drives the load
average past 30 (observed on a single pipeline run, RSS only 0.8 GB —
this is CPU oversubscription, not memory).

Setting these env vars *before* the numeric libraries are imported caps
each backend's thread pool.  We use a moderate value (4) rather than 1:
under the single-instance lock only one pipeline runs at a time, so a
handful of threads per backend keeps throughput up without
oversubscribing the cores.

``setdefault`` so an operator can still override from the environment
(e.g. ``OMP_NUM_THREADS=1 python -m src.pipeline.main ...``).
"""

import os
import warnings

_THREAD_CAP = os.environ.get("FFTM_THREAD_CAP", "4")

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",   # macOS Accelerate / vecLib
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, _THREAD_CAP)

# Suppress noisy third-party warnings that otherwise flood the log:
#  - google.auth/oauth2 FutureWarning (Python 3.9 EOL) printed once per
#    Gemini call → thousands of lines on a table-heavy run.  Matched by
#    source module (its message is multi-line, so a message regex needs
#    DOTALL and is fragile).
#  - paddle "No ccache found" UserWarning printed once per rec-predict.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\..*")
warnings.filterwarnings("ignore", message=r"(?s).*No ccache found.*")
