"""Chat templating, loss masking, and dataset prep.

The deck's claim (§13.2) is that loss masking and the chat template decide more
outcomes than every LoRA variant combined. This module exists so you can *see* the
mask rather than trust a library flag.

The masking technique used here renders the conversation to **text** around each
assistant turn, then maps character spans onto tokens:

    prefix = apply_chat_template(messages[:i], add_generation_prompt=True)   # text
    upto   = apply_chat_template(messages[:i+1])                             # text
    supervised characters = [len(prefix), len(upto))
    supervised tokens     = tokens whose offsets fall inside that range

**Why not just diff the token lists?** Because that is wrong on real templates, and
Qwen3.5 is the counter-example. Its prefix ends `<think>\n` while the full render
continues `\n</think>`; tokenized separately the trailing `\n` is its own token, but
tokenized together `\n\n` merges into a *different single token*. The token lists are
therefore not prefix-related even though the strings are. Diffing tokens raises a
false alarm at best and silently mis-masks at worst.

Characters do not have this problem, and `return_offsets_mapping` gives the mapping
back to tokens — including for special tokens like `<|im_end|>`, which must stay
supervised because it is the model's stop signal.
"""
from __future__ import annotations

import statistics
import warnings
from dataclasses import dataclass, field

IGNORE_INDEX = -100

# What the loss is computed over. Deck §13.5: on a reasoning base, this choice can
# preserve or destroy the model's reasoning behaviour, and the safe option is
# model-dependent — so it is a parameter, not a constant.
MASK_MODES = ("assistant-only", "masked-think", "response-only", "everything")


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    n_supervised: int
    n_total: int

    @property
    def supervised_fraction(self) -> float:
        return self.n_supervised / max(1, self.n_total)


class TemplateNotPrefixStable(RuntimeError):
    """Raised when apply_chat_template(msgs[:i+1]) does not start with the i-prefix.

    If you hit this, the template is rewriting earlier turns as the conversation grows
    (some templates move a system prompt, or strip reasoning from previous turns).
    Masking by difference is unsafe there — inspect the rendered strings before
    training, and see `thinking_survives()`.
    """


def _render(tokenizer, messages, add_generation_prompt: bool, enable_thinking: bool | None) -> str:
    """Render a conversation to text (never to tokens — see the module docstring)."""
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    if enable_thinking is not None:
        # Not every template accepts this; pass it only when asked for.
        kwargs["enable_thinking"] = enable_thinking
    out = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(out, list):
        out = out[0]
    return out


def _encode_with_offsets(tokenizer, text: str):
    """Tokenize rendered chat text, returning (ids, char_offsets).

    `add_special_tokens=False` because the template already emits them; this was
    verified to reproduce `apply_chat_template(tokenize=True)` exactly on Qwen3.5.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping")
    if offsets is None:
        raise RuntimeError(
            "This tokenizer did not return offset mappings (is it a 'slow' tokenizer?). "
            "Loss masking here needs a fast tokenizer: "
            "AutoTokenizer.from_pretrained(..., use_fast=True)."
        )
    return list(enc["input_ids"]), list(offsets)


def find_span(haystack: list[int], needle: list[int], start: int = 0) -> int:
    """Index of `needle` in `haystack` at/after `start`, or -1."""
    if not needle:
        return -1
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def build_example(
    tokenizer,
    messages: list[dict],
    max_length: int = 1024,
    mask_mode: str = "assistant-only",
    enable_thinking: bool | None = None,
    think_open: str = "<think>",
    think_close: str = "</think>",
) -> Example:
    """Tokenize a conversation and build the label mask.

    mask_mode:
      "assistant-only" — supervise every assistant token (the usual SFT default)
      "masked-think"   — supervise the assistant turn but NOT its reasoning block
      "response-only"  — supervise only what follows </think> (strictest)
      "everything"     — supervise all tokens, prompt included. This is the classic
                         bug (§16: "model writes your question back at you"); it is
                         selectable so NB1 can show you what it looks like.
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")

    full_text = _render(tokenizer, messages, False, enable_thinking)
    full_ids, offsets = _encode_with_offsets(tokenizer, full_text)
    labels = [IGNORE_INDEX] * len(full_ids)

    if mask_mode == "everything":
        labels = list(full_ids)
    else:
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            prefix_text = _render(tokenizer, messages[:i], True, enable_thinking)
            upto_text = _render(tokenizer, messages[:i + 1], False, enable_thinking)
            if not full_text.startswith(prefix_text):
                raise TemplateNotPrefixStable(
                    f"turn {i}: rendering messages[:{i}] with add_generation_prompt is not "
                    "a text prefix of the full render. This template rewrites earlier "
                    "turns as the conversation grows; masking by difference is unsafe."
                )
            start_char, end_char = len(prefix_text), len(upto_text)
            if mask_mode in ("masked-think", "response-only"):
                start_char = _skip_reasoning_chars(
                    full_text, start_char, end_char, think_close)
            for j, (a, b) in enumerate(offsets):
                if b > a and a >= start_char and b <= end_char:
                    labels[j] = full_ids[j]

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]

    return Example(
        input_ids=full_ids,
        labels=labels,
        n_supervised=sum(1 for x in labels if x != IGNORE_INDEX),
        n_total=len(full_ids),
    )


def _skip_reasoning_chars(text: str, start: int, end: int, think_close: str) -> int:
    """Move `start` past a closing reasoning tag inside text[start:end), if present.

    Character-based for the same reason as the main path: tag boundaries do not align
    with token boundaries.

    **When this actually fires.** Qwen3.5 does emit `<think>\n\n</think>\n\n` for a
    non-reasoning answer — but it emits it as part of the *generation prompt*, i.e.
    inside `prefix_text`. So `start` is already past `</think>` and there is nothing
    left in `[start, end)` to skip: on a corpus of plain answers `masked-think` and
    `response-only` produce a mask identical to `assistant-only`. This only does work
    when the assistant *content* carries its own reasoning block, which is what a
    §13.5 experiment needs its training data to look like.
    """
    at = text.find(think_close, start, end)
    if at == -1:
        return start           # no reasoning block in this turn — nothing to skip
    return at + len(think_close)


# --- inspection helpers: the point of NB1 -----------------------------------

def decode_supervised(tokenizer, ex: Example) -> str:
    """Exactly the text the loss is computed on. Read this before you train."""
    kept = [t for t, l in zip(ex.input_ids, ex.labels) if l != IGNORE_INDEX]
    return tokenizer.decode(kept, skip_special_tokens=False)


def decode_masked(tokenizer, ex: Example) -> str:
    """The complement: everything the model sees but is not scored on."""
    dropped = [t for t, l in zip(ex.input_ids, ex.labels) if l == IGNORE_INDEX]
    return tokenizer.decode(dropped, skip_special_tokens=False)


def thinking_survives(tokenizer, think_open="<think>", think_close="</think>") -> dict:
    """Deck §16: some chat templates DELETE reasoning blocks inside apply_chat_template.

    If that happens, the reasoning traces in your dataset never reach the loss — and
    nothing errors. Run this once per base model, before training.
    """
    body = "buoc 1: kiem tra. buoc 2: tra loi."
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": f"{think_open}{body}{think_close}4"},
    ]
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception as exc:                      # pragma: no cover - template variance
        return {
            "ok": False,
            "error": repr(exc),
            "rendered": None,
            "open_tag_present": False,
            "body_present": False,
            "verdict": f"TEMPLATE FAILED TO RENDER: {exc}",
        }
    if isinstance(rendered, list):
        rendered = rendered[0]
    return {
        "ok": body in rendered,
        "open_tag_present": think_open in rendered,
        "body_present": body in rendered,
        "rendered": rendered,
        "verdict": (
            "reasoning preserved — safe to train on traces"
            if body in rendered
            else "TEMPLATE STRIPS REASONING — your traces will never reach the loss"
        ),
    }


def token_stats(lengths: list[int]) -> dict:
    """p50/p95/p99 so `max_length` is a measurement, not a guess (deck §13)."""
    if not lengths:
        return {"n": 0}
    ordered = sorted(lengths)

    def pct(p: float) -> int:
        # Nearest-rank percentile: index = ceil(p * n) - 1. Chosen over
        # round(p*(n-1)) because Python's round() is banker's rounding, which makes
        # the median of 1..100 land on 51 instead of 50 — a silent off-by-one in the
        # number students use to set `max_length`.
        import math
        idx = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
        return ordered[idx]

    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 1),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": ordered[-1],
        "suggested_max_length": _round_pow2(pct(0.95)),
    }


def _round_pow2(n: int) -> int:
    p = 256
    while p < n and p < 32768:
        p *= 2
    return p


def to_messages(record: dict) -> list[dict]:
    """Normalize the common instruction formats to a messages list."""
    if "messages" in record and isinstance(record["messages"], list):
        return record["messages"]
    instruction = record.get("instruction") or record.get("prompt") or record.get("question") or ""
    context = record.get("input") or record.get("context") or ""
    output = record.get("output") or record.get("response") or record.get("answer") or ""
    user = f"{instruction}\n\n{context}".strip() if context else str(instruction).strip()
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": str(output).strip()},
    ]


def to_training_dataset(
    tokenizer,
    records: list[dict],
    max_length: int = 1024,
    mask_mode: str = "assistant-only",
    enable_thinking: bool | None = None,
    think_open: str = "<think>",
) -> list[dict]:
    """Pre-tokenize records into {input_ids, labels} using the mask NB1 verified.

    **Why pre-tokenize instead of letting TRL do it.** TRL's `assistant_only_loss`
    derives its mask from `{% generation %}` markers in the chat template. Qwen3.5's
    template has none, so that flag yields a mask of ZERO supervised tokens — and
    transformers only *warns*. Training runs to completion and the numbers are
    meaningless. Run `scripts/check_mask_agreement.py` to see it on your own base model.

    Training on the output of `build_example()` means the loss covers exactly the tokens
    the student decoded and asserted on in NB1. No flag in between to be wrong.

    Trade-off: pre-tokenized labels cannot be `packing`-ed (packing concatenates
    examples and would invalidate the label alignment). Correctness of the mask outranks
    the throughput here.
    """
    # `masked-think` and `response-only` differ from `assistant-only` only when the
    # assistant CONTENT carries its own reasoning block. Qwen3.5's generation prompt
    # already closes an empty `<think></think>`, so on a corpus of plain answers the
    # skip has nothing left to skip and all three modes emit an identical mask — see
    # `_skip_reasoning_chars`. The shipped triage corpus is exactly that: 250 bare-JSON
    # answers. A student who sets MASK_MODE for the §13.5 contrast would otherwise see
    # no difference and no reason why, so say it out loud instead of being silently inert.
    if mask_mode in ("masked-think", "response-only"):
        if not any(think_open in (r.get("output") or "") for r in records):
            warnings.warn(
                f"mask_mode={mask_mode!r} is a no-op on this corpus: none of the "
                f"{len(records)} records carry a {think_open} block in their answer, "
                "and the chat template closes its empty reasoning block inside the "
                "generation prompt. The resulting mask is identical to 'assistant-only'. "
                "Exercising deck §13.5 needs training answers that contain real traces "
                "— see 'Đổi dataset của riêng bạn' in README.md.",
                RuntimeWarning,
                stacklevel=2,
            )

    out: list[dict] = []
    for r in records:
        ex = build_example(
            tokenizer, to_messages(r), max_length=max_length,
            mask_mode=mask_mode, enable_thinking=enable_thinking,
        )
        if ex.n_supervised == 0:
            continue                       # nothing to learn from; drop it
        out.append({"input_ids": ex.input_ids, "labels": ex.labels,
                    "attention_mask": [1] * len(ex.input_ids)})
    if not out:
        raise RuntimeError(
            "every example ended up with zero supervised tokens — the mask is wrong. "
            "Re-run NB1 and read results/mask_proof.json before training."
        )
    return out


def split(records: list, train_frac: float = 0.9, seed: int = 42) -> tuple[list, list]:
    """Deterministic split. Same seed in every notebook so runs stay comparable."""
    import random
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    cut = int(len(idx) * train_frac)
    return [records[i] for i in idx[:cut]], [records[i] for i in idx[cut:]]
