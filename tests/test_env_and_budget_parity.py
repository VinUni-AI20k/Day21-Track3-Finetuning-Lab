"""Regression tests for the silent-configuration bugs.

Every case here is a failure that produced no error message: a `.env` that was written,
read by nobody, and obeyed by nothing; an autopsy whose contrasts trained for a
different number of steps than the baseline they are compared against; a verdict
decided by string-matching its own prose. They are grouped in one file because they
share that shape, not because they share a module.
"""
from __future__ import annotations

import os

import pytest

from labkit import env as labenv
from labkit import evaluate as ev
from labkit import train
from labkit.config import get_tier


# --- .env is actually read ---------------------------------------------------

def test_parse_handles_the_forms_dotenv_example_uses():
    parsed = labenv.parse(
        "# comment\n"
        "\n"
        "COMPUTE_TIER=LAPTOP\n"
        "export MASK_MODE=masked-think\n"
        'QUOTED="two words"\n'
        "SINGLE='x'\n"
        "EPOCHS=3   # trailing comment\n"
        "NOT_A_PAIR\n"
    )
    assert parsed == {
        "COMPUTE_TIER": "LAPTOP",
        "MASK_MODE": "masked-think",
        "QUOTED": "two words",
        "SINGLE": "x",
        "EPOCHS": "3",
    }


def test_dotenv_reaches_get_tier(tmp_path, monkeypatch):
    """The bug: README/HARDWARE-GUIDE/.env.example all promised this and it never worked."""
    monkeypatch.delenv("COMPUTE_TIER", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("COMPUTE_TIER=LAPTOP\n", encoding="utf-8")
    labenv.load_dotenv(dotenv)
    assert os.environ["COMPUTE_TIER"] == "LAPTOP"
    assert get_tier().name == "LAPTOP"


def test_explicit_environment_beats_the_file(tmp_path, monkeypatch):
    """`EVAL_LIMIT=8 make pipeline` and CI must override .env, not lose to it."""
    monkeypatch.setenv("COMPUTE_TIER", "BIGGPU")
    dotenv = tmp_path / ".env"
    dotenv.write_text("COMPUTE_TIER=CPU\n", encoding="utf-8")
    applied = labenv.load_dotenv(dotenv)
    assert applied == {}
    assert get_tier().name == "BIGGPU"


def test_missing_dotenv_is_not_an_error(tmp_path):
    assert labenv.load_dotenv(tmp_path / "nope.env") == {}


# --- NB3 and NB4 spend the same step budget ----------------------------------

@pytest.mark.parametrize("epochs", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("tier_name", ["CPU", "LAPTOP", "T4", "BIGGPU"])
def test_contrast_budget_tracks_the_epoch_knob(monkeypatch, epochs, tier_name):
    """F-17 regressed: the fix derived the contrast budget from a hardcoded 2.0 while
    NB3 kept reading $EPOCHS, so the two agreed only at the default. `.env.example`
    invites EPOCHS=1..3, and at 3 the contrasts trained for two-thirds of the baseline.
    """
    monkeypatch.setenv("EPOCHS", str(epochs))
    import importlib

    from labkit import config
    importlib.reload(config)
    try:
        tier = get_tier(tier_name)
        nb3 = train.planned_steps(225, tier, config.TRAIN_EPOCHS)
        nb4 = train.planned_steps(225, tier, config.CONTRAST_EPOCHS)
        assert nb3 == nb4, (
            f"{tier_name} @ EPOCHS={epochs}: baseline runs {nb3} steps, contrasts run "
            f"{nb4} — the autopsy would measure training length, not configuration"
        )
    finally:
        monkeypatch.delenv("EPOCHS", raising=False)
        importlib.reload(config)


def test_default_epoch_budget_is_still_two():
    from labkit import config
    assert config.TRAIN_EPOCHS == 2.0
    assert config.CONTRAST_EPOCHS == config.TRAIN_EPOCHS


# --- the verdict comes from the numbers, not from the prose ------------------

def _scores(target, regression):
    return ev.GroupScores(target=target, regression=regression, format=1.0,
                          latency_ms=1.0, n=10)


def test_verdict_survives_rewording_its_reasons(monkeypatch):
    """It used to be `not any(r.startswith(("target", "general")) ...)`. A graded
    pass/fail must not depend on the first word of a human-readable sentence."""
    base = _scores(0.50, 0.70)
    win = ev.regression_gate(_scores(0.80, 0.70), base)
    assert win.passed is True

    lost_target = ev.regression_gate(_scores(0.40, 0.70), base)
    assert lost_target.passed is False

    forgot = ev.regression_gate(_scores(0.80, 0.50), base)
    assert forgot.passed is False

    # A tie is not a win: MIN_TARGET_GAIN requires strictly better than baseline (b).
    assert ev.regression_gate(_scores(0.50, 0.70), base).passed is False


def test_small_regression_inside_tolerance_still_passes():
    """A run may trade a little general capability for target gain — that is what the
    tolerance is for. Deliberately not testing the exact knife-edge: `0.70 - 0.02`
    is 0.6799999999999999 in binary floating point, so an equality test there asserts
    something about float representation rather than about the gate.
    """
    base = _scores(0.50, 0.70)
    inside = ev.regression_gate(_scores(0.60, 0.70 - ev.REGRESSION_TOLERANCE / 2), base)
    assert inside.passed is True

    outside = ev.regression_gate(_scores(0.60, 0.70 - ev.REGRESSION_TOLERANCE * 2), base)
    assert outside.passed is False


# --- mask modes announce themselves when they cannot do anything -------------

def test_reasoningless_corpus_warns_for_think_modes():
    """The shipped corpus is 250 bare-JSON answers, so `masked-think` and
    `response-only` are byte-identical to `assistant-only` there. Silent no-ops are
    what this lab is about; this one now says so."""
    from labkit import data
    from tests.fake_tokenizer import FakeTokenizer

    records = [{"instruction": "i", "input": "x", "output": '{"a": 1}'}]
    with pytest.warns(RuntimeWarning, match="no-op on this corpus"):
        data.to_training_dataset(FakeTokenizer(), records, max_length=128,
                                 mask_mode="masked-think")


def test_no_warning_when_the_corpus_has_traces():
    from labkit import data
    from tests.fake_tokenizer import FakeTokenizer

    records = [{"instruction": "i", "input": "x",
                "output": "<think>vi sao</think>\n{\"a\": 1}"}]
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)
        data.to_training_dataset(FakeTokenizer(), records, max_length=128,
                                 mask_mode="masked-think")
