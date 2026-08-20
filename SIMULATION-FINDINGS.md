# Simulation findings — Lab 21

Student simulation of the documented path. Two environments:

| Env | Hardware | Python | Purpose |
|---|---|---|---|
| **local** | Apple M4, 16 GB, MPS (no CUDA) | 3.14.3 | fast reproduction + fixes |
| **Colab** | free-tier **Tesla T4 16 GB** | 3.x (Colab) | the documented default path |

Stack resolved on both (from PyPI, 2026-08-21):
`torch 2.13.0 · transformers 5.15.1 · trl 1.10.0 · peft 0.20.0 · accelerate 1.14.0 ·
datasets 5.0.1` — `bitsandbytes` correctly skipped on macOS by the platform marker.

---

## F-01 — NB1 crashed on the lab's own default model — **FIXED**

**Severity: critical.** The lab could not get past its first notebook on `Qwen3.5-4B`.

**Symptom.** `TemplateNotPrefixStable: turn 1: rendering messages[:2] does not extend
messages[:1]`. Raised on *ordinary* prompt→answer data, not an edge case.

**Root cause — subtle and worth reading.** `build_example()` diffed **token lists**
around each assistant turn:

```
generation prompt ends:  ...<|im_start|>assistant\n<think>\n
full render continues:   ...<think>\n\n</think>\n\n{answer}<|im_end|>\n
```

The *strings* are prefix-related (`full.startswith(prefix)` is `True`). The *token
lists* are not: the prefix's trailing `\n` is one token, but in the full render `\n\n`
merges into a single **different** token. Diffing tokens therefore compares
non-comparable sequences.

Two aggravating factors specific to 2026 reasoning models:
* Qwen3.5 emits `<think>\n\n</think>\n\n` **even when the answer contains no
  reasoning**, so every sample crossed this boundary.
* The template also *normalizes* author-supplied `<think>` blocks, so hand-written
  reasoning data renders differently from what you wrote.

**Fix.** Render to text, tokenize once with `return_offsets_mapping=True`, and
supervise tokens whose character span falls inside `[len(prefix), len(upto))`.
Verified: this reproduces `apply_chat_template(tokenize=True)` ids **exactly**, and
special tokens carry real offsets, so `<|im_end|>` — the stop signal — stays
supervised.

**Why the test suite missed it.** The fake tokenizer was a plain ChatML renderer with
character-level tokenization: no think scaffold, no token merging. It could not
express the failure. The fake now reproduces both behaviours, plus 5 regression tests
(58 → 63), including one that asserts the *fixture itself* still produces
non-prefix-related token lists — otherwise the regression test would pass vacuously.

---

## F-02 — `resolve_target_modules` picks up hybrid-attention projections — **BY DESIGN, documented**

On the real `Qwen3.5-4B`, `text-linear` resolves to **12** suffixes, not the 7 a plain
transformer has:

```
down_proj gate_proj up_proj  q_proj k_proj v_proj o_proj
in_proj_a in_proj_b in_proj_qkv in_proj_z out_proj   <- Gated DeltaNet layers
```

The extra five are the **linear-attention** layers (deck §6.4: 24 linear + 8 full
attention, a 3:1 interleave). Adapting them is correct — they are part of the text
decoder — but it changes the arithmetic: matched rank for attention-only is **r≈283**
on the real model versus r≈90 on a plain-transformer shape. `matched_rank()` computes
this at runtime, so the contrast stays fair. Confirmed the vision tower is still
excluded.

---

## F-03 — `AutoModelForCausalLM` accepts the multimodal config — **NO ACTION**

De-risked without downloading weights via `init_empty_weights()` +
`from_config`. `Qwen3_5Config` (architectures: `Qwen3_5ForConditionalGeneration`)
loads as `Qwen3_5ForCausalLM`. `generate.load_base()` works on the default tier.

---

## F-04 — `transformers` does not depend on `jinja2` — **FIXED (pre-Colab)**

`apply_chat_template` raises `ImportError` without it. Would have broken NB1 on cell
one for every student. Pinned in both requirements files.

---

## F-05 — Colab free tier allows only ONE GPU session — **DOCUMENTED**

Opening a second lab notebook while the first holds a runtime gives *"Quá nhiều phiên
đang hoạt động"* and the second silently never starts. Students running NB1 in one tab
and NB3 in another will hit this. Belongs in HARDWARE-GUIDE.md.

---

## F-06 — a "16 GB" T4 gives **14.6 GB** usable — **DOC FIX NEEDED**

Colab reports `VRAM: 14.6 GB`, not 16. The `Qwen3.5-4B` bf16 checkpoint is **9.32 GB**
on the wire, so weights alone take ~62% of the card before any activations, LoRA state
or optimizer moments. HARDWARE-GUIDE.md says "Colab Free T4 16 GB" and budgets ~10 GB
for 4B bf16 LoRA — the headroom is thinner than documented. Whether it actually fits
is what NB3 decides; the number in the guide should say 14.6 GB either way.

---

## F-07 — the lab hardcoded **bf16**, and its default GPU has none — **FIXED**

**Severity: high.** Found by inspection during the Colab run, then fixed pre-emptively.

`sft_config_kwargs()` emitted `bf16=True` unconditionally and `load_base()` used
`dtype=torch.bfloat16`. The lab's **default tier is a free-Colab T4**, which is Turing
(sm_75) — **bfloat16 requires Ampere (8.0+)**. So the recommended path was configured
for hardware the recommended hardware is not.

This is the standard 2026 tutorial bug: every guide is written on an A100, where
`bf16=True` is correct, and the flag gets copied onto cards that cannot do it.

fp16 is not a drop-in swap either: its exponent range is far smaller, so training needs
**gradient scaling** to avoid underflow — which trainers enable only when told
`fp16=True`. The precision decision therefore has to reach the *training arguments*,
not just the model load.

**Fix.** New `labkit/device.py`: `describe()` / `precision()` / `torch_dtype()` /
`banner()` pick bf16 → fp16 → fp32 from the actual device capability, and
`sft_config_kwargs(precision=...)` allows an explicit override. Both flags are always
set, never both true. 7 new tests (63 → 70), including one asserting the flags are
never hardcoded and one that a T4-shaped device produces the explanatory note.

---

## F-08 — NB2/NB5 print nothing for tens of minutes — **FIXED**

**Severity: medium (usability, but it makes students kill good runs).**

`score_run()` prints only after a whole baseline finishes. On the free T4 the observed
gap between "Loading weights: 100%" and the first number was **>15 minutes** with zero
output. That is indistinguishable from a hang, and the documented remedy for a hung
Colab cell is to interrupt it.

**Fix.** `generate_batch()` now prints a per-batch line with elapsed time and ETA,
labelled by which pass is running (`(a) base + naive prompt/target`, `ft/regression`,
…). NB2 and NB5 pass labels through.

## F-09 — the published time budget is optimistic — **DOC FIX NEEDED**

README claims NB2 ≈ 10 min and the core ≈ 80 min on a T4. Measured on free Colab:

| Stage | Claimed | Observed |
|---|---|---|
| NB1 | 2 min | **26 s** ✅ |
| model download (first run only) | not mentioned | **~70 s** for 9.32 GB |
| weight load | not mentioned | ~30 s |
| NB2 baseline (a) alone | — | **>15 min** (pre-fp16) |

**Measured after the fp16 fix**, from the new progress output:

```
[(a) base + naive prompt/target] batch 1/13    44s elapsed  ~523s left
[(a) base + naive prompt/target] batch 2/13    88s elapsed  ~483s left
```

**44 s per batch of 4 prompts ≈ 11 s/prompt** at `max_new_tokens=160`, 4B on a T4.
Extrapolating:

| Stage | README claim | Projected from measurement |
|---|---|---|
| NB1 | 2 min | **14–26 s** ✅ |
| NB2 (two baselines × 65 prompts) | 10 min | **≈ 23 min** |
| NB5 (one scoring pass) | 10 min | **≈ 12 min** |
| **core NB1–NB5** | **80 min** | **≈ 95–110 min** |

The structural cause is that the eval set is generated **three times** across the lab
(baseline a, baseline b, fine-tune). That is inherent to the three-baseline design and
is the right trade — but the README must say so, and `EVAL_LIMIT` should be presented
as the normal way to iterate rather than a hidden knob.

---

## F-10 — `assistant_only_loss=True` supervises **ZERO tokens** on the default model — **CRITICAL**

The single most damaging finding, and a **silent** one.

NB3 configured `assistant_only_loss=True` and handed training to TRL. TRL derives that
mask from `{% generation %}` markers in the chat template. **Qwen3.5's template has
none.** Result, measured by `scripts/check_mask_agreement.py`:

```
chat template exposes {% generation %} markers: False
labkit assistant-only : 11/31 tokens (35.5%)   '</think>\n\n{"intent": "doi_tra"}<|im_end|>\n'
TRL  assistant_masks  :  0/31 tokens ( 0.0%)   ''
VERDICT: FAIL — TRL would supervise NOTHING.
```

transformers emits a **warning, not an error**. Training completes. A loss curve is
drawn. The numbers are meaningless.

This is precisely the class of bug the deck spends §13.2 and §16 on — *"no error, a
plausible loss curve, and a broken model"* — reproduced by the lab's own default
configuration. NB1 proves the mask is correct and then NB3 threw that proof away and
trusted a library flag.

**Fix.** Stop trusting the flag. NB3 now trains on the **exact mask NB1 verified**:
`data.to_training_dataset()` pre-tokenizes with `build_example()`, so `input_ids` and
`labels` are the ones the student decoded and asserted on. `assistant_only_loss` is not
set at all.

Consequence, stated honestly on the slide-facing side: pre-tokenized labels are
incompatible with `packing`, so packing is off for this path. Deck §13.3's point
(packing is free only when boundaries are respected) still stands — here the *mask's
correctness* outranks the throughput, and the lab says so rather than quietly keeping a
flag that does nothing.

`scripts/check_mask_agreement.py` ships with the lab so students can run this check
against any base model they swap in.

---

## F-11 — the format scorer was stricter than the target scorer — **FIXED**

`triage_field_accuracy()` recovered a `{...}` block embedded in prose;
`has_required_keys()` accepted only bare or fenced JSON. A model answering
`"Day la ket qua: {...}"` therefore scored on **target** but **0.000 on format** — a
formatting failure that did not happen. Two scorers disagreeing about what counts as
JSON makes both numbers untrustworthy, and `format` is one of the four graded groups.

Both now share `_parse_json_loose()`. +2 tests (76 total).

Surfaced by the first real T4 measurement: `(a) base + naive prompt  target=0.000
format=0.000`. Those particular zeros turned out to be genuine — a naive prompt with no
schema produces prose, not JSON — but checking *why* they were zero exposed the
inconsistency.

## F-12 — observation: the optimized prompt is ~3× faster, not just more accurate

Measured on the T4: baseline (a) ran at **44 s/batch**, baseline (b) at **15 s/batch**.
Same model, same prompts, same decode settings. The optimized prompt tells the model to
emit only JSON, so it emits ~20 tokens and stops; the naive prompt lets it ramble to the
160-token cap.

Worth teaching: prompt engineering bought a **3× latency win before any fine-tuning**,
which sharpens deck §17's point that baseline (b) is a real bar — it is better on the
target metric *and* cheaper to serve.

---

## F-13 — Colab's preinstalled **torchao 0.10.0** blocks NB3 entirely — **FIXED**

**Severity: high — this is a hard stop on the documented default path.**

```
ImportError: Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported
```

Raised inside `get_peft_model()` → `_create_new_module()`. Colab preinstalls
torchao 0.10.0; peft 0.20 / transformers 5.15 require >0.16. **Nothing in
`requirements.txt` pulled torchao in**, so pip never upgraded it and the stale
preinstalled copy won. NB3 failed after 51 s.

This is the failure mode a locally-tested lab cannot catch: the machine that broke it
is the one with *extra* packages already installed, not missing ones. `pip install -r`
succeeds; the conflict only appears at import time inside a third-party call.

**Fix.** Explicit `torchao>=0.16` in `requirements.txt` and in both Colab bootstraps,
with a comment saying why a package nothing imports directly is pinned.

### Also confirmed on Colab: F-10, independently

The same cell ran `scripts/check_mask_agreement.py` on the real T4:

```
labkit assistant-only : 11/31 tokens (35.5%)
TRL  assistant_masks  :  0/31 tokens ( 0.0%)
VERDICT: FAIL — TRL would supervise NOTHING.
```

Identical to the local reproduction — the F-10 fix is aimed at a real defect on the
real platform, not an artifact of the Mac.

---

## F-14 — `padding_free=True` was unconditional; on the default tier it is both unsafe and useless — **FIXED**

NB3 died at trainer construction:

```
ValueError: When `padding_free=True` without packing, `max_length` is not enforced.
```

preceded by two warnings that matter more than the error:

```
Padding-free training is enabled, but the attention implementation is not set to a
supported Flash Attention variant ... only the following are known to reliably support
this: flash_attention_2, flash_attention_3, ...
Using a batch size of 1 annihilates the benefits of padding-free training.
```

Three separate problems in one flag:

1. **Unsafe here.** Padding-free flattens a batch into one sequence. Without a kernel
   that understands the boundaries, attention can run *across* them — literally deck
   §13.3's warning ("packing is free only when sequence boundaries are respected")
   applied to packing's sibling flag. **FlashAttention-2 needs Ampere (sm_80+), so a T4
   cannot have it at all.**
2. **Useless here.** The T4 tier uses `per_device_train_batch_size=1`. There is no
   inter-sequence padding to remove in a batch of one.
3. **Incompatible with the F-10 fix.** Pre-tokenized labels force `packing=False`, and
   TRL rejects `padding_free` + `max_length` without packing.

**Fix.** `device.supports_padding_free(batch)` requires an importable FlashAttention
kernel **and** batch ≥ 2. When it *is* available, `max_length=None` is passed —
honest, because `build_example()` already truncates — rather than silencing the check.
+3 tests (79 total).

**Meta-point worth keeping:** the deck teaches `packing` + `padding_free` as the §13.3
recommendation. On the hardware the lab actually recommends, neither is available. The
lab now says that out loud instead of setting flags that do not apply.

---

## F-15 — my own F-07 fix was wrong: `torch.cuda.is_bf16_supported()` returns **True on a T4** — **FIXED**

Caught by reading NB3's config dump during the training run: `"bf16": "True",
"fp16": "False"` — on a T4, after supposedly fixing exactly this.

The trap is in torch's signature:

```python
def is_bf16_supported(including_emulation: bool = True):
    ...
    if torch.cuda.get_device_properties(device).major >= 8:
        return True
    if not including_emulation:
        return False
    return _check_bf16_tensor_supported(device)     # <- Turing passes this
```

**The default is `including_emulation=True`**, and it only checks that a bf16 tensor can
be *created*. Turing can, by emulation. So the "is bf16 supported?" API answers **yes**
on hardware with no bf16 units, and training proceeds emulated at a large speed
penalty while truthfully reporting `bf16=True`.

This very likely explains F-09's slow generation: the whole first NB2 run was emulated.

**Fix.** Compute capability ≥ 8.0 is the real test, with
`is_bf16_supported(including_emulation=False)` as a secondary check where the kwarg
exists. +1 test that fakes a T4 (`capability 7.5`, `is_bf16_supported() → True`) and
asserts we still choose fp16. **80 tests.**

**The lesson worth carrying:** a capability API that answers "yes, by emulation" is
worse than no API. Three of this lab's eighteen findings (F-10, F-15, F-16) are the same
shape — a library saying *yes*, or saying nothing much, about something it is not
really doing.

---

## F-16 — `warmup_ratio` no longer exists, so every run trained with **no warmup** — **FIXED**

Visible in NB3's own config dump, one line above the hyper-parameters:

```
⚠ TRL không nhận: ['warmup_ratio']
```

`filter_kwargs` did its job and said so. Nobody read it — including me, across a full
NB3 run.

transformers v5 / TRL 1.10 removed the field. Measured on the Colab VM:

```python
[f.name for f in dataclasses.fields(SFTConfig) if "warm" in f.name]
# -> ['warmup_steps']          # no ratio field at all
```

So `sft_config_kwargs` asked for a knob that does not exist, the filter dropped it with a
warning, and training ran with a cosine schedule and **zero warmup** while printing a
config the reader would assume included it.

**Fix.** Convert the deck's 10% into an absolute step count via a new
`train.planned_steps()`, and emit `warmup_steps`. +3 tests.

**The lesson.** F-16 is the *cost* of the version-defensive design that F-10 justified.
Dropping unknown kwargs converts a loud `TypeError` into a quiet behaviour change; the
warning is only a safeguard if something fails when it fires. A stricter contract —
"warn, and fail if the dropped key is one the recipe depends on" — would have caught
this at step 0 instead of after a 25-minute run.

---

## F-17 — the contrasts were trained **twice as long** as the baseline they are compared against — **FIXED**

`config.py` stated the requirement and then broke it:

```python
# Every contrast run gets the SAME number of optimizer steps as the NB3 baseline
# slice, otherwise the comparison measures wall-clock, not configuration.
CONTRAST_MAX_STEPS = 60
```

NB3 does not run 60 steps. It runs an *epoch* budget — 2 epochs × ⌈225/16⌉ — which the
real T4 run printed as `100% 30/30`. So each of NB4's three contrasts got **2× the
training** of the `correct` run it is measured against, and NB4's own prose had already
absorbed the bug as a workaround:

> `correct` từ NB3 chạy nhiều step hơn — **đừng so loss trực tiếp với nó** … hãy chạy
> lại `correct` với `max_steps=CONTRAST_MAX_STEPS` (một dòng, **~10 phút**)

Two things wrong there. The comparison the notebook is *for* was being deferred to an
optional manual re-run; and "~10 phút" is the estimate the constant was sized against —
measured at 48.5 s/step, 60 steps is **48 minutes** per contrast, so NB4 alone was a
**~145 minute** stage inside a lab advertised at ~80.

**Fix.** Both sides derive the budget from the same recipe
(`train.planned_steps(len(train_ds), TIER, CONTRAST_EPOCHS)`), so the autopsy varies one
variable instead of two, the manual fix-up disappears, and NB4 drops to ~73 min. +2 tests.

**Caught by arithmetic, not by running it** — NB4 had never executed. The measured
48.5 s/step from the one completed NB3 run is what made the 60 visibly wrong.

---

## F-18 — the generated Colab notebooks were stale and still shipped the F-13 crash — **FIXED**

`colab/*.ipynb` is generated from `notebooks/*.py` by `scripts/build_colab.py`. The F-13
fix added `torchao>=0.16` to that script's BOOTSTRAP — but the notebooks were never
regenerated and committed, so all six still carried the old bootstrap.

Consequence: the RUN_ALL path was fixed, while the **per-notebook Colab badges in the
README** — the entry point a student following the lab notebook-by-notebook actually
uses — still walked into the torchao 0.10 `ImportError` at `get_peft_model()`.

Found only because F-17 forced a regeneration and the diff showed an unrelated line
changing. **Generated artifacts that are committed need a build step in the gate**, or
they drift silently from their source.

---

## F-19 — Colab never re-reads the notebook, so a long-lived tab runs *old* code — **ENVIRONMENT, not the repo**

Cost me an 8-minute pipeline run and looked exactly like a regression.

The restarted run died at NB3 with the F-13 error — `Found an incompatible version of
torchao. Found version 0.10.0` — a bug fixed two days earlier, in a VM where cell 1 had
just run green.

Cell 1 was **stale**. Colab fetches notebook source from GitHub *once*, when the URL is
opened, and never again: not on reconnect, not on a new runtime, not when the repo moves.
This tab was opened in the previous session, before `82bda58` added the pin, so the cell
it ran was the pre-fix list. It installed the seven packages it knew about, reported
success in 4 s, and left `torchao 0.10.0` in place.

Three symptoms that make this hard to read correctly:

* the cell **succeeds** — nothing in its output hints the source is old
* `git pull` inside the cell updates the *repo*, which makes the environment look fresh
  while the cell doing the pulling is itself out of date
* the printed `commit :` line reports the freshly pulled HEAD — **a stale cell can print
  a current commit hash**, which is actively misleading

**Not a repo defect** — the committed notebook is correct, and any student opening the
badge gets it. Worth a README line anyway, because "reconnect and re-run" is the natural
reaction to a disconnect and it silently preserves the stale source. The reliable move
after any repo change is to reload the browser tab, not just the runtime.

---

## F-20 — the dependency list existed in **three** hand-synced copies — **FIXED**

`requirements.txt`, `scripts/build_colab.py`'s BOOTSTRAP, and `colab/Lab21_RUN_ALL.ipynb`
cell 1 each carried their own copy of the same pins. Keeping them in sync was manual, and
it had already failed twice:

* **F-18** — BOOTSTRAP got `torchao>=0.16`, the generated notebooks did not
* **F-13's recurrence** — the pin reached `requirements.txt` and the two bootstraps on
  different days, which is what made F-19 possible at all

The failure mode is nasty because a bootstrap missing a pin **does not fail at install
time**. It exits 0, and the run dies ten minutes later inside `get_peft_model()`, with a
traceback pointing at peft rather than at the install cell that actually caused it.

**Fix.** Both bootstraps now `pip install -q -r requirements.txt`. The repo is cloned
before the install, so the file is available; torch is preinstalled on Colab and
`requirements.txt` pins it compatibly, so that line is a no-op. One source of truth.

---

## F-21 — the README's Quick Start pointed at a notebook that does not exist — **FIXED**

```
### Colab (khuyến nghị)
Mở `colab/Lab21_T4.ipynb` → Runtime → Change runtime type → T4 GPU → Run all.
```

There is no `colab/Lab21_T4.ipynb`. The directory holds `Lab21_01`..`Lab21_06` and
`Lab21_RUN_ALL`. The **first instruction in the lab** named a file that was never
generated — a leftover from an earlier naming scheme that no test covers, because nothing
verifies that documentation references resolve.

Found by grepping the README for "colab" while fixing something else, not by any check.

**Fix.** Point at `Lab21_RUN_ALL.ipynb` as a clickable Colab link, and fold the F-19
reload warning in next to it — that is where a student is standing when it bites them.

---

## F-22 — `.env` was never read by anything — **FIXED**

**Severity: high.** README, `HARDWARE-GUIDE.md`, `.env.example` and `config.py`'s own
docstring all state that editing `.env` selects the tier. Nothing parsed the file.
`get_tier()` consulted `os.environ` only, and no dependency or code path populated it:

```
$ grep -c dotenv requirements*.txt src/labkit/*.py     # 0
$ echo COMPUTE_TIER=LAPTOP > .env && python notebooks/01_data_and_mask.py
tier=T4  model=unsloth/Qwen3.5-4B          # <- not LAPTOP
```

`MASK_MODE` and `EPOCHS` were ignored the same way. A student on an 8-12 GB laptop who
follows the documented instruction gets the T4 tier's 4B model and OOMs ten minutes
later, having done exactly what they were told. Invisible on Colab, because there the
documented default and the real default happen to agree.

**Fix.** `labkit/env.py` — a dependency-free `.env` parser loaded from
`labkit/__init__.py` before any submodule reads the environment. An already-set
variable always wins, so `EVAL_LIMIT=8 make pipeline` and CI still override the file.
+4 tests.

---

## F-23 — `make smoke` failed on the GPU-less path it exists to serve — **FIXED**

**Severity: high (it is the first command a student without a GPU runs).**

`tests/test_modeling_and_train.py` imported `torch` unguarded inside one test.
`requirements-cpu.txt` ships no torch *by design* — it is the slice advertised as
"enough for NB1 + the whole test suite, no GPU".

```
$ make setup-cpu && make smoke
[ FAIL ] unit tests    1 failed, 88 passed
Not ready to submit — fix the FAILs above.
```

The failing test is F-15's own regression test, so the CPU slice was broken by the fix
for a GPU bug. **Fix.** `pytest.importorskip("torch")`. Verified in a torch-free venv:
`109 passed, 1 skipped`, exit 0.

---

## F-24 — F-17's fix only held at the default `EPOCHS` — **FIXED**

**Severity: high (silently invalidates the comparison the lab is graded on).**

F-17 replaced `CONTRAST_MAX_STEPS = 60` with a derived budget so that "the autopsy
varies one variable instead of two". The replacement derived it from a hardcoded
`CONTRAST_EPOCHS = 2.0` while NB3 kept reading `$EPOCHS` — so the two agreed only at
the default, and `.env.example` invites `EPOCHS=1..3`:

```
EPOCHS=1: NB3=29 steps, contrasts=58     <- contrasts trained 2x the baseline
EPOCHS=2: NB3=58 steps, contrasts=58
EPOCHS=3: NB3=87 steps, contrasts=58     <- contrasts trained 0.67x the baseline
```

Exactly the defect F-17 describes, reintroduced by anyone who touched the knob the
docs offer them. `verify.py` never noticed: it checked the *parameter* budget of
`attn_only` and never the *step* budget of anything.

**Fix.** One value, `config.TRAIN_EPOCHS`, with `CONTRAST_EPOCHS` as an alias; NB3
imports it instead of re-reading the environment. NB3 now records `max_steps` in
`runs.csv`, and `verify.py` gained a step-parity gate that FAILs when the runs
disagree. +9 tests.

---

## F-25 — `colab_run.py` block-buffered the child's stdout, defeating F-08 — **FIXED**

**Severity: medium.** The docstring claims "Output is NOT captured, so Colab streams it
live and a long training run does not look like a hang." The children were spawned with
neither `-u` nor `PYTHONUNBUFFERED`, so CPython block-buffers stdout whenever it is a
**pipe** — which is what Colab, `tee`, and every redirect hand the child.

Measured through a pipe, first line out of a 3-line script that prints once per 1.2 s:

```
before fix: first line reached the pipe after 3.61s   (i.e. only at process exit)
after  fix: first line reached the pipe after 0.01s
```

Observed live: >3 minutes of total stdout silence during NB2 while stderr flowed
normally. The casualties are F-08's per-batch ETA lines — added specifically so students
stop killing healthy runs — and they never reached the student. **Fix.** `-u` plus
`PYTHONUNBUFFERED=1` on the child.

---

## F-26 — `verify.py` green-lit smoke runs — **FIXED**

**Severity: medium.** NB2 already writes `smoke_mode` and `eval_limit` into
`baselines_frozen.json`, and `.env.example` states that a submitted run must leave
`EVAL_LIMIT` unset. `verify.py` read neither key, so an 8-item run printed
`Ready to submit.` **Fix.** A `full eval set used` check that FAILs on `smoke_mode`,
naming the item count and how to re-run.

> Follow-up for the maintainer: `colab/Lab21_RUN_ALL.ipynb` defaults its widget to
> `EVAL_LIMIT = "8"`, so the documented one-click path now produces a run this gate
> correctly rejects. Left as-is deliberately — whether the default Colab run should be
> submittable (~95-110 min) or fast (~15 min) is a product call, not a bug fix.

---

## F-27 — NB6 scored on 20 items in a full run — **FIXED**

**Severity: low.** `notebooks/06_merge_and_serve.py` defaulted `EVAL_LIMIT` to `"20"`
while every other notebook uses `0` = full set, so the merge no-regression assert
silently ran on 20 of 50 items even in an unabridged run. **Fix.** Default `0`, slice
only when set, and print the count.

---

## F-28 — the graded verdict was decided by string-matching its own prose — **FIXED**

**Severity: low (correct today, one reworded sentence from being wrong).**

```python
passed=not any(r.startswith(("target", "general")) for r in reasons)
```

`regression_gate` computes both numeric conditions, throws them away, formats them into
human-readable sentences, and then recovers the verdict by inspecting the first word of
those sentences. Rewording a message — or translating it, in a lab written in
Vietnamese — silently flips a pass to a fail. **Fix.** Derive `passed` from the
booleans. +2 tests.

---

## F-29 — `MASK_MODE` is inert on the shipped corpus, and a comment said otherwise — **FIXED (documented)**

**Severity: medium (pedagogical, not behavioural).**

`masked-think` and `response-only` produce a mask **identical** to `assistant-only` on
the shipped data — 37/188 supervised tokens for all three. Two independent reasons:

1. All 250 training answers are bare JSON; `<think>` appears zero times across all four
   `data/*.jsonl` files.
2. More subtly, `add_generation_prompt=True` already emits the *complete* empty block:

```
prefix: '...<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

so the supervised span starts *past* `</think>` and `_skip_reasoning_chars` has nothing
left to skip. Its docstring claimed the opposite — "Qwen3.5 emits `<think>\n\n</think>\n\n`
even for a non-reasoning answer, so this fires on ordinary data too". It emits it, but
into the prefix, so the skip never fires on ordinary data.

Knock-on: NB5's `valid_trace_rate` is structurally 0.0 for every run — the model is
never trained on traces and generation runs `enable_thinking=False`.

**Fix.** Corrected the docstring, documented the caveat in `.env.example`, and made
`to_training_dataset()` warn when a think-mode is selected against a corpus that cannot
exercise it. The knob is not silently inert any more. Deliberately **not** fixed by
adding traces to the corpus: that would change `data/checksums.json` and trip the
eval-drift gate for everyone. +2 tests.

---

## Verified working

| Check | Where | Result |
|---|---|---|
| `git clone` + `pip install` bootstrap | Colab T4 | ✅ `GPU: Tesla T4`, ~30 s |
| GitHub → Colab notebook launch | Colab | ✅ renders, Vietnamese intact |
| Tokenizer download + chat template | both | ✅ |
| `thinking_survives()` on real template | both | ✅ "reasoning preserved" |
| NB1 end-to-end, real 4B tokenizer | local | ✅ 39/188 supervised, both asserts green |
| Unit tests | both | ✅ 63 passed |
| `requirements.txt` resolution | local (py3.14) | ✅ dry-run clean |
| `verify.py` smoke + full | **Colab T4** | ✅ incl. integrity checks on real artifacts: `baseline (b) beats (a) 0.000 -> 0.760`, prompt SHA unmodified, eval checksums intact, unfilled REPORT template correctly FAILED |
| **NB2 end-to-end, real 4B on T4** | Colab T4 | ✅ **1006 s**, `baselines_frozen.json` written |
| **NB3 config + dataset build** | Colab T4 | ✅ 12 target modules resolved, 32.46 M trainable, matched rank r=283 (budgets within 0.03%), 225 rows, **9014/42101 tokens supervised (21.4%)**, assert passed |
| **NB3 training started** | Colab T4 | ✅ **30 steps, 48 s/step** — reached `1/30 [00:48<23:18]` before the browser extension disconnected |

---

## Measured results (free Colab T4, `unsloth/Qwen3.5-4B`, full 50-item eval)

| Run | target | regression | format | latency |
|---|---|---|---|---|
| **(a)** base + naive prompt | 0.000 | 0.724 | 0.000 | 11331 ms |
| **(b)** base + optimized prompt | **0.760** | 0.724 | **1.000** | **3775 ms** |
| (c) LoRA fine-tune | *not reached* | | | |

**The lab's central design validated itself empirically.** Baseline (b) is a genuinely
hard bar — 0.760 target with perfect JSON compliance and 3× lower latency than (a).
A fine-tune has to beat *that*, which is exactly the discipline deck §17 argues for and
the opposite of the old lab's perplexity-vs-nothing comparison.

Deck §6.4 also became a lab artifact: NB3 printed the real model's
`layer_types: {linear_attention: 24, full_attention: 8}` — the 3:1 hybrid interleave,
read off the checkpoint the student is fine-tuning.

---

## Not verified

The browser extension disconnected mid-training, so these remain **unrun**:

* **NB3 completion** — it was training (30 steps, 48 s/step, ~23 min ETA) with the
  post-fix configuration when the connection dropped. Everything up to and including
  the first optimizer step is verified.
* **NB4** (three contrast runs) and **NB5** (verdict) — never reached.
* The **fp16** path end-to-end: NB3's observed run used *emulated* bf16 because of
  F-15, which was fixed after that run started. The fix is unit-tested but has not
  itself been exercised on a T4.

**To resume** (the Colab runtime and `results/` survive; artifacts are on disk there):

```
!pip install -q "torchao>=0.16" && git pull -q && python scripts/colab_run.py nb3 nb4 nb5
```

Nothing in the resume depends on this machine — the repo has every fix.
