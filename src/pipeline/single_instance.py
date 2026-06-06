"""Cross-process single-instance lock for the heavy pipeline.

The unified pipeline loads MolNexTR (~1 GB), PaddleOCR, TATR and two
YOLO models per run; a single run can peak near the machine's physical
RAM.  Launching two runs concurrently doubles that and drives the box
into heavy swap (observed: load average spiking past 12, UI freeze).

This module guards against that: ``single_instance_lock`` takes an
exclusive advisory lock on a lockfile.  If another pipeline already
holds it, the second invocation refuses to start and exits cleanly with
a clear message rather than piling on memory pressure.

The lock is advisory (``fcntl.flock``) and released automatically when
the process exits — even on crash — because the OS drops the fd's lock.
"""

from __future__ import annotations

import errno
import fcntl
import os
import sys
from contextlib import contextmanager
from typing import Iterator

_DEFAULT_LOCK_DIR = os.path.join("data", ".locks")


@contextmanager
def single_instance_lock(name: str, lock_dir: str = _DEFAULT_LOCK_DIR) -> Iterator[None]:
    """Hold an exclusive advisory lock for the duration of the ``with`` block.

    Parameters
    ----------
    name:
        Logical lock name (one lockfile per name).  Use a stable string
        like ``"flowfigtabminer-pipeline"`` so every pipeline invocation
        contends for the same lock.
    lock_dir:
        Directory to place the lockfile in.  Created if missing.

    Behaviour
    ---------
    If the lock is free, yields immediately and releases on exit.  If
    another process holds it, prints who holds it and exits the process
    with status 1 — the heavy pipeline must never run two-up.
    """
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{name}.lock")

    # Open (not truncate) so a stale file's previous PID stays readable
    # until we win the lock and overwrite it.
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                holder = _read_holder(fd)
                print(
                    f"[single-instance] Another pipeline is already running "
                    f"{holder}.\n"
                    f"[single-instance] Refusing to start a second run — it would "
                    f"double memory use and swap-thrash the machine.\n"
                    f"[single-instance] Wait for it to finish, or kill it, then retry.",
                    file=sys.stderr,
                )
                os.close(fd)
                sys.exit(1)
            raise

        # We hold the lock — stamp our identity for the next contender.
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.fsync(fd)

        try:
            yield
        finally:
            # Releasing the lock explicitly; the fd close would also do it.
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _read_holder(fd: int) -> str:
    """Best-effort read of the lockfile's stamped PID for the message."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, 256).decode(errors="replace").strip()
        return f"({data})" if data else "(pid unknown)"
    except OSError:
        return "(pid unknown)"
