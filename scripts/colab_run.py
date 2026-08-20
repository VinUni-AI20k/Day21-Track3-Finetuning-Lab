#!/usr/bin/env python3
"""Run lab stages back-to-back with live output. Built for Colab, useful anywhere.

    python scripts/colab_run.py nb1 nb2 nb3 nb4 nb5
    python scripts/colab_run.py all
    EVAL_LIMIT=8 python scripts/colab_run.py nb2 nb5     # smoke mode

Output is NOT captured, so Colab streams it live and a long training run does not look
like a hang. Stops at the first failing stage and reports which one.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

STAGES = {
    "nb1": ("notebooks/01_data_and_mask.py", "data, chat template & loss mask"),
    "nb2": ("notebooks/02_baselines.py", "freeze eval + three baselines"),
    "nb3": ("notebooks/03_train_correct.py", "train the correct configuration"),
    "nb4": ("notebooks/04_misconfig_autopsy.py", "three misconfiguration contrasts"),
    "nb5": ("notebooks/05_evaluate_and_verdict.py", "four-group eval + verdict"),
    "nb6": ("notebooks/06_merge_and_serve.py", "merge + adapter hot-swap (optional)"),
}
CORE = ["nb1", "nb2", "nb3", "nb4", "nb5"]


def main(argv: list[str]) -> int:
    wanted = argv or CORE
    if wanted == ["all"]:
        wanted = CORE + ["nb6"]
    bad = [w for w in wanted if w not in STAGES]
    if bad:
        print(f"unknown stage(s) {bad}; pick from {list(STAGES)} or 'all'", file=sys.stderr)
        return 2

    print(f"tier={os.environ.get('COMPUTE_TIER', 'T4')}  "
          f"mask={os.environ.get('MASK_MODE', 'assistant-only')}  "
          f"eval_limit={os.environ.get('EVAL_LIMIT') or 'full'}")

    timings: list[tuple[str, float]] = []
    for name in wanted:
        script, desc = STAGES[name]
        print("\n" + "=" * 72, flush=True)
        print(f"{name.upper()} — {desc}", flush=True)
        print("=" * 72, flush=True)
        t0 = time.perf_counter()
        # `-u` is load-bearing, not decoration. Without it CPython block-buffers stdout
        # whenever it is a pipe -- which is what Colab, `tee`, and any redirect give the
        # child -- so `print()` output sits in an 8 KB buffer while stderr flows freely.
        # That silences exactly the per-batch ETA lines F-08 added to stop students
        # killing healthy runs, and reproduces the hang-that-isn't this file's docstring
        # claims to prevent. Observed: >3 minutes of stdout silence during NB2.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        rc = subprocess.run([sys.executable, "-u", script], cwd=ROOT, env=env).returncode
        dt = time.perf_counter() - t0
        timings.append((name, dt))
        if rc != 0:
            print(f"\n!! {name.upper()} FAILED (exit {rc}) after {dt:.0f}s — stopping.",
                  flush=True)
            _summary(timings)
            return rc
        print(f"\n-- {name.upper()} ok in {dt:.0f}s", flush=True)

    _summary(timings)
    return 0


def _summary(timings: list[tuple[str, float]]) -> None:
    print("\n" + "-" * 40)
    print("stage timings")
    for name, dt in timings:
        print(f"  {name:5s} {dt:7.0f}s  ({dt/60:.1f} min)")
    print(f"  {'total':5s} {sum(d for _, d in timings):7.0f}s "
          f"({sum(d for _, d in timings)/60:.1f} min)")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
