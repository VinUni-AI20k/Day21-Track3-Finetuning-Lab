"""Structural checks — the class of defect the behaviour tests are blind to.

Three findings in this lab (F-18, F-20, F-21) were all the same shape: a *derived or
referenced* artifact drifting from its source with nothing to catch it.

  F-18  colab/*.ipynb drifted from the build_colab.py BOOTSTRAP that generates them
  F-20  the dependency pins existed in three hand-synced copies
  F-21  the README's first instruction named a notebook that has never existed

None of them are behaviour bugs, so 84 behaviour tests said nothing. They cost zero
runtime to check.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_verify_checksums_ignore_windows_line_endings(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "verify_script", ROOT / "scripts" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    lf.write_bytes(b'{"row":1}\n{"row":2}\n')
    crlf.write_bytes(b'{"row":1}\r\n{"row":2}\r\n')

    assert mod._sha(lf) == mod._sha(crlf)


def _bootstrap() -> str:
    """BOOTSTRAP out of scripts/build_colab.py (not a package — load it by path)."""
    spec = importlib.util.spec_from_file_location(
        "build_colab", ROOT / "scripts" / "build_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BOOTSTRAP


# --- F-21: every path the docs name must exist -------------------------------
PATH_RE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:ipynb|py|md|txt|jsonl|sh))`")

# Paths the docs tell the STUDENT to create. Absent from the repo by design — but listed
# explicitly so a genuine typo in one of them still fails.
STUDENT_CREATED = {"data/CUSTOM_DATASET.md"}


@pytest.mark.parametrize("doc", ["README.md", "HARDWARE-GUIDE.md", "rubric.md"])
def test_documented_paths_exist(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = []
    for ref in set(PATH_RE.findall(text)):
        if "/" not in ref:                 # bare filenames are prose, not paths
            continue
        if ref.startswith(("http", "adapters/", "results/", "submission/", "data/split")):
            continue                        # produced by a run, not shipped
        if ref in STUDENT_CREATED:
            continue
        # `labkit/device.py` is how you refer to the module; it lives under src/
        if not ((ROOT / ref).exists() or (ROOT / "src" / ref).exists()):
            missing.append(ref)
    assert not missing, f"{doc} names files that do not exist: {sorted(missing)}"


# --- F-18: generated notebooks must match their generator --------------------
def test_generated_notebooks_carry_the_current_bootstrap():
    boot = _bootstrap()
    stale = []
    for nb_path in sorted((ROOT / "colab").glob("Lab21_0*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        if "".join(nb["cells"][0]["source"]).strip() != boot.strip():
            stale.append(nb_path.name)
    assert not stale, (
        f"regenerate with scripts/build_colab.py — stale bootstrap in {stale}")


# --- F-20: one source of truth for dependency pins ---------------------------
def test_bootstraps_install_from_requirements_not_a_copied_list():
    run_all = json.loads((ROOT / "colab" / "Lab21_RUN_ALL.ipynb").read_text(encoding="utf-8"))
    sources = {
        "build_colab.BOOTSTRAP": _bootstrap(),
        "Lab21_RUN_ALL cell 1": "".join(run_all["cells"][1]["source"]),
    }
    for name, src in sources.items():
        assert "requirements.txt" in src, f"{name} does not install from requirements.txt"
        # a copied list re-pins packages inline; that is exactly the drift F-20 fixed
        inline = re.findall(r'"(transformers|trl|peft|accelerate|datasets|torchao)>=', src)
        assert not inline, f"{name} re-pins {sorted(set(inline))} inline — drift risk"


# --- F-22: the autopsy must be settled on the task metric, not on train loss ---------

NB4 = ROOT / "notebooks" / "04_misconfig_autopsy.py"
NB5 = ROOT / "notebooks" / "05_evaluate_and_verdict.py"


def test_nb5_scores_every_contrast_adapter():
    """NB4 trains three misconfigured adapters, saves them, and reports `final_loss` --
    a TRAINING loss. The lab names exactly that substitution "Lỗi #3". Until this fix,
    nothing ever scored those adapters on the target task, so NB4's question "did the
    mistake lose?" was answered by the proxy metric the lab warns against."""
    src = NB5.read_text(encoding="utf-8")
    assert "CONTRAST_KEYS" in src, "NB5 must iterate the contrast runs, not just `correct`"
    assert "autopsy.json" in src, "the per-adapter task scores must be written to an artifact"


def test_contrast_adapters_are_scored_at_their_training_precision():
    """`qlora` learned against a 4-bit base. Scoring it on the fp16 base measures a
    base/adapter mismatch and labels the damage 'what QLoRA costs you'."""
    src = NB5.read_text(encoding="utf-8")
    assert "load_in_4bit=SPECS[key].load_in_4bit" in src, (
        "take the 4-bit flag from the run's own spec, never from score_adapter's default")


def test_nb4_does_not_present_train_loss_as_the_verdict():
    """The column can stay -- it is free and it is interesting. It just may not be the
    thing the student ranks the four runs by."""
    src = NB4.read_text(encoding="utf-8")
    assert "final_loss" in src
    assert "LOSS HUẤN LUYỆN" in src, "the column must be labelled as a training loss"
    assert "NB5" in src, "NB4 must point at the notebook that actually settles the ranking"


def _notebook_text(nb_path):
    import json
    raw = json.loads(nb_path.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", [])) for c in raw.get("cells", []))


@pytest.mark.parametrize("src", sorted((ROOT / "notebooks").glob("*.py")),
                         ids=lambda p: p.name)
def test_generated_notebook_matches_its_source(src):
    """`colab/*.ipynb` is generated from `notebooks/*.py`, and nothing enforced that the
    generated copy was rebuilt after the source changed. F-18 is exactly that bug: the
    notebooks shipped a crash already fixed in the source, so every Colab student hit it
    while the repo looked correct. Checking only the bootstrap cell was not enough.
    """
    nb = ROOT / "colab" / f"Lab21_{src.stem}.ipynb"
    assert nb.exists(), f"{nb.name} missing — run `make colab`"
    generated = _notebook_text(nb)

    missing = [
        line for line in src.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip())
        and not stripped.startswith("#")        # markdown/comments move around
        and len(stripped) > 12                  # skip `import os`, `rows = []`, ...
        and stripped not in generated
    ]
    assert not missing, (
        f"{nb.name} is stale — {len(missing)} source lines are not in it. "
        f"Run `make colab`. First: {missing[0][:90]!r}")


# --- F-31: the prompt trained on must be the prompt evaluated with -------------------

def test_training_prompt_is_the_evaluation_prompt():
    """Measured on a T4: every adapter scored target=0.000 format=0.000 with a healthy
    loss curve, because training folded the schema into the user turn while evaluation
    sent `NAIVE_PROMPT` + a bare ticket. The model answered the question it was actually
    asked -- vague instruction, no schema -- in fluent prose.

    A mask proof does not catch this: the mask was correct throughout. What was wrong is
    what the model was conditioned on.
    """
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    from fake_tokenizer import FakeTokenizer

    from labkit import data
    from labkit.config import NAIVE_PROMPT

    tok = FakeTokenizer()
    rec = {"instruction": "Classify the ticket into JSON with 4 keys.",
           "input": "my order VN1 is late", "output": '{"a": 1}'}

    good = data.prompt_alignment(tok, rec, system=NAIVE_PROMPT)
    assert good["eval_prompt_is_prefix_of_training"], (
        "the text generation sends is not a prefix of what training saw — the fine-tune "
        "is being asked to continue a prompt it never trained on")
    assert '{"a": 1}' in good["supervised_tail"]

    # The old shape must still be detectable as broken, or this gate proves nothing.
    broken = data.prompt_alignment(tok, rec, system=None)
    eval_text = good["eval_render"]
    assert not broken["train_render"].startswith(eval_text), (
        "the pre-F-31 shape should NOT align with the evaluation prompt")


def test_prompts_have_exactly_one_definition():
    """generate.py re-exports them; config.py owns them. Two copies is how F-31 got in."""
    from labkit import config, generate
    assert generate.NAIVE_PROMPT is config.NAIVE_PROMPT
    assert generate.OPTIMIZED_PROMPT is config.OPTIMIZED_PROMPT
    gen_src = (ROOT / "src" / "labkit" / "generate.py").read_text(encoding="utf-8")
    assert "NAIVE_PROMPT = " not in gen_src, "generate.py must import, not redefine"
