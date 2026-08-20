#!/usr/bin/env python3
"""Pre-submission gatekeeper for Lab 21.

Run `python scripts/verify.py` before you submit. It checks that the artifacts exist,
that they are internally consistent, and that the *comparison you are graded on was
actually a fair one*.

Modes:
  --smoke   imports, tier resolution, seed data present, unit tests. No GPU, no model.
  (default) everything above plus the artifact + integrity checks.

Exit code 0 = ready to submit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((status, name, detail))


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# --- smoke ------------------------------------------------------------------

def smoke() -> None:
    try:
        from labkit import config, data, evaluate, modeling, report, train  # noqa: F401
        check("labkit imports", OK)
    except Exception as exc:
        check("labkit imports", FAIL, repr(exc))
        return

    from labkit.config import TIERS, get_tier
    try:
        tier = get_tier()
        check("tier resolves", OK, f"{tier.name} -> {tier.model_id}")
    except Exception as exc:
        check("tier resolves", FAIL, repr(exc))

    # NOTE: a for/else here would ALWAYS run the else branch (it fires when the loop
    # ends without `break`), reporting PASS even after a FAIL. Use an explicit flag.
    offenders = [n for n, t in TIERS.items() if t.effective_batch > 32]
    if offenders:
        check("tier batch rule", FAIL,
              f"{offenders} exceed the 32 effective-batch ceiling (deck §10.4)")
    else:
        check("all tiers respect the <32 effective-batch rule", OK)

    for f in ("train_seed.jsonl", "eval_target.jsonl", "eval_regression.jsonl"):
        p = ROOT / "data" / f
        check(f"data/{f}", OK if p.exists() else FAIL,
              f"{sum(1 for _ in p.open(encoding='utf-8'))} rows" if p.exists()
              else "missing — run scripts/make_seed_data.py")

    # NOTE: pyproject sets addopts="-q". Passing -q again yields -qq, which hides the
    # summary line. Use -rN + --tb=no and let the configured -q stand.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "--tb=no", "-rN"],
        capture_output=True, text=True, cwd=ROOT)
    lines = [l.strip() for l in (proc.stdout or proc.stderr).splitlines() if l.strip()]
    # The last line may be pytest's progress dots; the summary is the line that reports
    # counts. Search backwards for it rather than assuming position.
    summary = next(
        (l for l in reversed(lines)
         if ("passed" in l or "failed" in l or "error" in l) and not set(l) <= set(". %[]0123456789")),
        lines[-1] if lines else f"exit {proc.returncode}",
    )
    check("unit tests", OK if proc.returncode == 0 else FAIL, summary)


# --- full -------------------------------------------------------------------

REQUIRED_ARTIFACTS = {
    "results/template_check.json": "NB1 — does the chat template keep <think>?",
    "results/mask_proof.json": "NB1 — proof the loss mask is right",
    "results/token_stats.json": "NB1 — p95 -> max_length",
    "results/baselines_frozen.json": "NB2 — the three-baseline table, frozen",
    "results/runs.csv": "NB3/NB4 — one row per training run",
    "results/verdict.json": "NB5 — the regression-gate verdict",
    "submission/REPORT.md": "your evaluation report",
}


def full() -> None:
    for rel, why in REQUIRED_ARTIFACTS.items():
        p = ROOT / rel
        check(f"{rel}", OK if p.exists() else FAIL, "" if p.exists() else why)

    # The shipped REPORT.md is a TEMPLATE. Existing is not the same as filled in.
    rep = ROOT / "submission" / "REPORT.md"
    if rep.exists():
        text = rep.read_text(encoding="utf-8")
        placeholders = text.count("<điền>") + text.count("<paste>") + text.count("<0.xx>")
        if placeholders:
            check("REPORT.md filled in", FAIL,
                  f"{placeholders} placeholders still present (<điền>, <paste>, ...) — "
                  "this is still the template")
        elif len(text.split()) < 400:
            check("REPORT.md filled in", WARN,
                  f"only ~{len(text.split())} words; the rubric asks for a ≥150-word "
                  "conclusion plus a ≥100-word verdict reading")
        else:
            check("REPORT.md filled in", OK, f"~{len(text.split())} words")

    # --- NB1 integrity ---
    proof = _load_json(ROOT / "results" / "mask_proof.json")
    if proof:
        good = proof.get("answer_is_supervised") and proof.get("question_is_masked")
        check("mask proof asserts", OK if good else FAIL,
              "" if good else "the loss mask is wrong — re-run NB1 before training")
        frac = proof.get("supervised_fraction", 0)
        if frac >= 0.95:
            check("supervised fraction", FAIL,
                  f"{frac:.0%} of tokens are in the loss — you are training on the prompt too")

    # --- NB2 integrity: was baseline (b) a real bar? ---
    frozen = _load_json(ROOT / "results" / "baselines_frozen.json")
    if frozen:
        # NB2 already records this; nothing read it, so `verify.py` cheerfully printed
        # "Ready to submit." for a run scored on 8 items. `.env.example` says a
        # submitted run must leave EVAL_LIMIT unset — this is the check that enforces it.
        if frozen.get("smoke_mode"):
            check("full eval set used", FAIL,
                  f"EVAL_LIMIT={frozen.get('eval_limit')} — this run scored "
                  f"{frozen.get('n_target')} target items, not the full set. Smoke mode "
                  "is for iterating; unset EVAL_LIMIT and re-run NB2+NB5 to submit.")
        else:
            check("full eval set used", OK, f"{frozen.get('n_target')} target items")

        try:
            from labkit.generate import OPTIMIZED_PROMPT
            expected = hashlib.sha256(OPTIMIZED_PROMPT.encode()).hexdigest()[:16]
            same = frozen.get("optimized_prompt_sha") == expected
            check("baseline (b) prompt unmodified", OK if same else WARN,
                  "" if same else
                  "OPTIMIZED_PROMPT differs from the shipped one. That is allowed only if "
                  "you made it STRONGER — say so in REPORT.md; weakening it to flatter the "
                  "fine-tune is the main way to fail this lab's honesty check.")
        except Exception as exc:
            check("baseline (b) prompt check", WARN, repr(exc))

        a = frozen.get("baseline_a", {}).get("target")
        b = frozen.get("baseline_b", {}).get("target")
        if a is not None and b is not None:
            if b <= a:
                check("baseline (b) beats (a)", WARN,
                      f"(b)={b:.3f} <= (a)={a:.3f} — your 'optimized' prompt is not "
                      "actually better. Improve it before claiming a fine-tune win.")
            else:
                check("baseline (b) beats (a)", OK, f"(a)={a:.3f} -> (b)={b:.3f}")

    # --- eval set untouched ---
    declared = (ROOT / "data" / "CUSTOM_DATASET.md").exists()
    ref = _load_json(ROOT / "data" / "checksums.json")
    if ref:
        drift = [f for f, h in ref.items() if (ROOT / "data" / f).exists()
                 and _sha(ROOT / "data" / f) != h]
        if not drift:
            check("eval sets unmodified", OK)
        elif declared:
            check("eval sets changed", WARN,
                  f"{drift} differ, but data/CUSTOM_DATASET.md declares a custom corpus — OK")
        else:
            check("eval sets unmodified", FAIL,
                  f"{drift} changed. Editing the eval set after seeing results invalidates "
                  "the comparison. If you swapped corpus on purpose, add data/CUSTOM_DATASET.md.")

    # --- runs ---
    runs_csv = ROOT / "results" / "runs.csv"
    if runs_csv.exists():
        import csv
        rows = list(csv.DictReader(runs_csv.open(encoding="utf-8")))
        names = {r.get("run") for r in rows}
        check("NB3 run present", OK if "correct" in names else FAIL,
              "" if "correct" in names else "no `correct` row in results/runs.csv")
        missing = {"attn_only", "wrong_lr", "qlora"} - names
        check("NB4 contrast runs", OK if not missing else WARN,
              "" if not missing else f"missing {sorted(missing)} (core requires all three)")

        # Step parity. The parameter-budget check below proves `attn_only` compares
        # PLACEMENT rather than budget; this one proves every run compares
        # CONFIGURATION rather than training length. Both are needed: a contrast
        # trained for twice as many steps is not a contrast, and that is precisely the
        # bug F-17 describes.
        budgets = {r["run"]: int(r["max_steps"]) for r in rows
                   if r.get("run") and str(r.get("max_steps") or "").isdigit()}
        if len(budgets) >= 2:
            if len(set(budgets.values())) == 1:
                check("all runs share one step budget", OK,
                      f"{sorted(budgets)} @ {next(iter(budgets.values()))} steps")
            else:
                check("all runs share one step budget", FAIL,
                      f"{budgets} — the autopsy measures training length, not "
                      "configuration. Did NB3 and NB4 see the same $EPOCHS?")
        elif rows:
            check("step budget recorded", WARN,
                  "runs.csv has no max_steps column — re-run NB3/NB4 so step parity "
                  "can be checked")

        for r in rows:
            if r.get("run") == "attn_only" and r.get("trainable_params") and r.get("r"):
                correct = next((x for x in rows if x.get("run") == "correct"), None)
                if correct and correct.get("trainable_params"):
                    got, want = int(r["trainable_params"]), int(correct["trainable_params"])
                    fair = abs(got - want) / max(1, want) < 0.05
                    check("attn_only is a FAIR contrast", OK if fair else FAIL,
                          f"{got:,} vs {want:,} trainable params"
                          + ("" if fair else " — budgets differ >5%, so this compares "
                                             "budget, not placement. Use matched_rank()."))

    # --- verdict ---
    verdict = _load_json(ROOT / "results" / "verdict.json")
    if verdict and "verdict" in verdict:
        v = verdict["verdict"]
        check("verdict recorded", OK,
              f"{'PASSED' if v.get('passed') else 'FAILED'} "
              f"(target Δ {v.get('target_delta', 0):+.3f}, "
              f"regression Δ {v.get('regression_delta', 0):+.3f})")
        if not v.get("passed"):
            check("note", WARN,
                  "A FAILED verdict is fully gradeable — analyse it honestly in REPORT.md "
                  "rather than loosening the gate.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="imports + data + tests only")
    args = ap.parse_args()

    smoke()
    if not args.smoke:
        full()

    width = max(len(n) for _, n, _ in results) + 2
    print()
    for status, name, detail in results:
        mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
        print(f"[{mark}] {name:<{width}} {detail}")

    fails = sum(1 for s, _, _ in results if s == FAIL)
    warns = sum(1 for s, _, _ in results if s == WARN)
    print(f"\n{len(results) - fails - warns} passed · {warns} warnings · {fails} failures")
    if fails:
        print("\nNot ready to submit — fix the FAILs above.")
        return 1
    print("\nReady to submit." + (" Read the warnings first." if warns else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
