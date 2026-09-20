"""Per-source outcome records — the one place a stage says why a source
lived or died.

Every stage that discards, skips or finishes a figure/table source calls
``write_status``.  The record lands at
``{intermediate_dir}/status/{source_id}.json`` and is the durable answer
to "where did this record die?" — previously that answer existed only in
stdout.  Writes are best-effort: a status failure must never abort the
stage that reports it.
"""

from __future__ import annotations

import json
import os
import time


def write_status(
    intermediate_dir: str,
    source_id: str,
    stage: str,
    outcome: str,
    reason: str = "",
    **extra,
) -> None:
    """Record ``outcome`` (``ok`` | ``filtered`` | ``failed``) for one source.

    ``stage`` names the pipeline stage reporting (``macro_clean``,
    ``coord_map``, ``table_filter``, ``table_vlm``, ``assembly`` …).
    A later call for the same source overwrites the earlier one, so the
    file always reflects the furthest stage reached.
    """
    if not intermediate_dir or not source_id:
        return
    try:
        status_dir = os.path.join(intermediate_dir, "status")
        os.makedirs(status_dir, exist_ok=True)
        record = {
            "source_id": source_id,
            "stage": stage,
            "outcome": outcome,
            "reason": reason,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        record.update(extra)
        with open(os.path.join(status_dir, f"{source_id}.json"), "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    except Exception as exc:  # never let bookkeeping break extraction
        print(f"      [Status] write failed for {source_id}: {exc}")
