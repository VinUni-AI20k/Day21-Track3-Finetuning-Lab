"""Load `.env` into the process environment.

**Why this file exists.** The README, `HARDWARE-GUIDE.md` and `.env.example` all tell
the student that editing `.env` selects the tier, the mask mode and the epoch budget.
Nothing read that file. `config.get_tier()` consulted `os.environ` only, so a student
who set `COMPUTE_TIER=LAPTOP` still got the T4 default — and on an 8 GB laptop GPU the
T4 tier means a 4B model and an OOM ten minutes later. The failure is silent and it is
invisible on Colab, because there the documented default and the real default agree.

Deliberately dependency-free: `requirements-cpu.txt` is the slice every student can
install with no GPU and no wheels to build, and `python-dotenv` is not worth a line in
it for thirty lines of parsing.

**Precedence — an already-set variable always wins.** `EVAL_LIMIT=8 make pipeline`,
`COMPUTE_TIER=CPU python ...` and CI environments must override the file, not lose to
it. That is the same rule `python-dotenv` applies with `override=False`.
"""
from __future__ import annotations

import os
import pathlib

_LOADED = False


def find_dotenv(start: pathlib.Path | None = None) -> pathlib.Path | None:
    """Nearest `.env` at or above `start` (default: this file's repo root)."""
    if start is None:
        start = pathlib.Path(__file__).resolve().parents[2]
    for directory in [start, *start.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def parse(text: str) -> dict[str, str]:
    """Parse `KEY=value` lines. Supports `export KEY=`, `#` comments and quoting.

    Kept small on purpose — this is the subset `.env.example` actually uses. Anything
    fancier belongs in python-dotenv, and if the lab ever needs it, take the dependency
    rather than growing this.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        out[key] = value
    return out


def load_dotenv(path: pathlib.Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Read `.env` into `os.environ`. Returns the variables it actually set."""
    path = path or find_dotenv()
    if path is None or not path.is_file():
        return {}
    applied: dict[str, str] = {}
    for key, value in parse(path.read_text(encoding="utf-8")).items():
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


def load_once() -> dict[str, str]:
    """Idempotent `load_dotenv()` — called on `import labkit`."""
    global _LOADED
    if _LOADED:
        return {}
    _LOADED = True
    return load_dotenv()
