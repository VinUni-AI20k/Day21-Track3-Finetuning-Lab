# %% [markdown]
# # NB6 (tuỳ chọn) — Merge, kiểm chứng sau merge, và hoán đổi adapter
#
# Deck §18. Hai đường triển khai:
# * **Merge** — `W = W₀ + (α/r)·BA`, đồ thị phục vụ giống hệt base → **không** overhead.
# * **Giữ riêng** — một base trong VRAM, nhiều adapter, chọn theo từng request.
#
# > Ngày 20 sở hữu tầng phục vụ (vLLM/SGLang). Ở đây chỉ làm phần **quyết định lúc huấn
# > luyện ảnh hưởng gì tới lúc phục vụ** — rồi bàn giao.

# %%
import json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

from labkit import evaluate as ev, generate, report
from labkit.config import get_tier

ROOT = pathlib.Path.cwd() if (pathlib.Path.cwd() / "data").exists() else pathlib.Path.cwd().parent
TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# EVAL_LIMIT=0 (unset) means the FULL set here, exactly as it does in NB2 and NB5.
# It used to default to 20, so an unabridged run silently scored this notebook's
# no-regression assert on 20 of 50 items while every other notebook used all of them.
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "0") or 0)
target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
if EVAL_LIMIT:
    target = target[:EVAL_LIMIT]
print(f"scoring merge check on {len(target)} target items")

# %% [markdown]
# ## 1. Điểm TRƯỚC merge

# %%
from peft import PeftModel

model, tok = generate.load_base(TIER)
model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / "correct"))
preds, _ = generate.generate_batch(model, tok, [r["input"] for r in target],
                                   system=generate.NAIVE_PROMPT)
before = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds, target)) / len(target)
print(f"trước merge: {before:.4f}")

# %% [markdown]
# ## 2. Merge và đo lại
#
# **Assert bắt buộc.** Merge là phép toán chính xác về mặt lý thuyết, nhưng dtype khi
# gộp có thể làm tụt điểm. Nếu tụt, đừng deploy — hãy tìm hiểu vì sao.

# %%
merged = model.merge_and_unload()
preds_m, _ = generate.generate_batch(merged, tok, [r["input"] for r in target],
                                     system=generate.NAIVE_PROMPT)
after = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds_m, target)) / len(target)
delta = after - before
print(f"sau merge:   {after:.4f}   (Δ {delta:+.4f})")

TOL = 0.01
assert delta >= -TOL, (
    f"điểm TỤT {abs(delta):.4f} sau merge (ngưỡng {TOL}). Kiểm tra dtype lúc merge; "
    "với DoRA cần PEFT ≥ 0.10 để gộp đúng vector magnitude (deck §18)."
)

out = ROOT / "adapters" / "merged"
merged.save_pretrained(out); tok.save_pretrained(out)
report.write_json({"before_merge": before, "after_merge": after, "delta": delta,
                   "tolerance": TOL, "n": len(target)},
                  "merge_check.json", results_dir=ROOT / "results")
del merged; generate.free_memory()

# %% [markdown]
# ## 3. Một base, nhiều adapter — hoán đổi theo request
#
# Đây là lập luận kinh tế của LoRA ở deck §18: base nằm trong VRAM một lần, mỗi khách
# hàng/tác vụ là một adapter vài chục MB.

# %%
model, tok = generate.load_base(TIER)
model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / "correct"),
                                  adapter_name="correct")
available = ["correct"]
for extra in ("attn_only", "qlora"):
    d = ROOT / "adapters" / extra
    if d.exists():
        model.load_adapter(str(d), adapter_name=extra)
        available.append(extra)

print("adapter đang nạp:", available)
ticket = target[0]["input"]
for name in available:
    model.set_adapter(name)
    out, _ = generate.generate_batch(model, tok, [ticket], system=generate.NAIVE_PROMPT)
    print(f"\n[{name}] -> {out[0][:140]}")

# %% [markdown]
# ## ✅ Checkpoint NB6
# - [ ] `results/merge_check.json` — điểm sau merge không tụt quá 0.01
# - [ ] Đã hoán đổi ≥2 adapter trên **cùng một** base đang nạp
#
# → Ngày 20 (Model Serving) là nơi biến việc này thành một endpoint đa người thuê.
