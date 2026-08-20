"""labkit — the shared harness for Day 21's fine-tuning lab.

Import order mirrors the notebooks:
    config    -> which model, which tier, which LoRA spec
    data      -> chat template, loss masking, dataset prep      (NB1, CPU-only)
    evaluate  -> four-group scoring + the regression gate        (NB2, NB5)
    modeling  -> where adapters go, and what they cost           (NB3, NB4)
    train     -> version-defensive TRL configuration             (NB3, NB4)
    report    -> results I/O so runs stay comparable

`.env` is loaded here, before any submodule reads `os.environ`, so that the file the
documentation tells students to edit actually reaches `config.get_tier()`. Variables
already present in the environment win — see `labkit.env`.
"""
from . import env as _env

_env.load_once()

__all__ = ["config", "data", "env", "evaluate", "modeling", "train", "report"]
__version__ = "1.0.0"
