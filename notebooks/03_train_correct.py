# %% [markdown]
# # NB3 — Huấn luyện cấu hình ĐÚNG
#
# Cấu hình ở đây là "vùng không hối tiếc" của deck §10, viết thẳng thành code:
#
# | Nút | Giá trị | Deck |
# |---|---|---|
# | `target_modules` | **toàn bộ linear của text decoder** | §10.2 |
# | `learning_rate` | **≈10× LR full-FT** | §10.3 |
# | batch hiệu dụng | **< 32** | §10.4 |
# | `packing` | **tắt** — xem ghi chú | §13.3 |
# | `padding_free` | chỉ khi có flash-attn **và** batch ≥ 2 | §13.3 |
#
# > **Vì sao khác deck §15.** Deck khuyến nghị bật `packing` + `padding_free`. Trên
# > model mặc định của lab, cả hai đều **không dùng được**, và lab nói thẳng thay vì
# > bật cờ vô tác dụng:
# >
# > * `packing` **tắt** vì ta nạp nhãn đã token hoá sẵn (mask đã kiểm chứng ở NB1).
# >   Packing nối các mẫu lại và sẽ phá vỡ căn chỉnh nhãn. *Tính đúng của mask quan
# >   trọng hơn thông lượng.*
# > * `padding_free` cần kernel **FlashAttention**, mà FA-2 đòi **Ampere (sm_80+)** —
# >   T4 là Turing, không bao giờ có. Và với `batch=1` thì cũng chẳng có padding nào
# >   để bỏ. Xem `labkit/device.py`.
# >
# > Tinh thần §13.3 vẫn đúng: *tăng tốc chỉ miễn phí khi ranh giới chuỗi được tôn
# > trọng.* Ở đây điều kiện đó không thoả, nên ta không bật.
# | `loss_type` | `chunked_nll` | §15 |
# | `alpha` | `2r` | §9.3 |

# %%
import json, os, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

from labkit import data, device, generate, modeling, report, train
from labkit.config import SPECS, TRAIN_EPOCHS, get_tier

ROOT = pathlib.Path.cwd() if (pathlib.Path.cwd() / "data").exists() else pathlib.Path.cwd().parent
TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
SPEC = SPECS["correct"]
print(f"{TIER.name} · {TIER.model_id} · {SPEC.label}")
print(device.banner())      # which precision is ACTUALLY being used, and why

# %% [markdown]
# ## 1. Nạp model — và nhìn vào kiến trúc bạn đang fine-tune
#
# Deck §6.4 nói các base 2026 xen kẽ **linear attention** với **full attention**. Đây là
# chỗ điều đó thôi là slide: config của chính model sẽ nói cho bạn biết.

# %%
model, tok = generate.load_base(TIER, load_in_4bit=SPEC.load_in_4bit)
print(json.dumps(modeling.layer_type_summary(model.config), ensure_ascii=False, indent=2))

# %% [markdown]
# ## 2. `all-linear` — nhưng không phải *mọi* linear
#
# Qwen3.5 là model **đa phương thức**: text decoder + vision tower. `target_modules=
# "all-linear"` của PEFT sẽ gắn adapter vào **cả vision encoder** bạn không hề huấn
# luyện — adapter phình to, step chậm hơn, và merge ra một checkpoint sai.
#
# `resolve_target_modules` trả về đúng phần text decoder.

# %%
targets = modeling.resolve_target_modules(model, SPEC.target)
trainable = modeling.count_lora_params(model, targets, SPEC.r)
print(f"placement={SPEC.target}  modules={targets}")
print(f"trainable LoRA params ≈ {trainable/1e6:.2f} M")

for row in modeling.describe_placement(model, SPEC.r):
    print("   ", row)

# %% [markdown]
# ## 3. Dataset đã tokenize + mask (dùng lại NB1)

# %%
from datasets import Dataset

split_dir = ROOT / "data" / "split"
assert split_dir.exists(), "Chạy NB1 trước — chưa có data/split/"

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

train_rows = load_jsonl(split_dir / "train.jsonl")
MASK_MODE = os.environ.get("MASK_MODE", "assistant-only")

# Train on the mask you PROVED in NB1 — not on a library flag.
#
# TRL's `assistant_only_loss` builds its mask from `{% generation %}` markers in the
# chat template. Qwen3.5 has none, so that flag supervises ZERO tokens while emitting
# only a warning: training completes, the loss curve looks fine, the run is worthless.
# Check it yourself:  python scripts/check_mask_agreement.py
rows = data.to_training_dataset(tok, train_rows, max_length=TIER.max_length,
                                mask_mode=MASK_MODE)
train_ds = Dataset.from_list(rows)
sup = sum(sum(1 for x in r["labels"] if x != data.IGNORE_INDEX) for r in rows)
tot = sum(len(r["labels"]) for r in rows)
print(train_ds)
print(f"mask_mode = {MASK_MODE}   supervised {sup}/{tot} tokens ({sup/tot:.1%})")
assert 0 < sup < tot, "mask covers nothing or everything — stop and re-run NB1"

# %% [markdown]
# ## 4. Cấu hình — lọc theo phiên bản TRL đang cài
#
# `filter_kwargs` hỏi TRL xem nó nhận tham số nào và **bỏ những gì không nhận, có cảnh
# báo**. Lab cũ phải monkey-patch `tokenizer=` → `processing_class=` và
# `evaluation_strategy` → `eval_strategy`; những bản vá đó là hoá thạch của TRL trước
# 1.0. Cách này không tạo ra hoá thạch mới.

# %%
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# Imported, not re-read from the environment: NB4 derives its contrast step budget
# from this same value, and two independent readings of $EPOCHS is exactly how the
# contrasts drifted to a different budget than the baseline they are compared with.
EPOCHS = TRAIN_EPOCHS
# NB4 runs its contrasts at exactly this many steps -- see labkit.config.CONTRAST_EPOCHS.
# Recorded in results/runs.csv so `verify.py` can prove the budgets matched, rather
# than asking the reader to trust that they did.
PLANNED_STEPS = train.planned_steps(len(rows), TIER, EPOCHS)
print(f"planned optimizer steps: {PLANNED_STEPS}  ({EPOCHS} epochs x {len(rows)} rows)")

want_sft = train.sft_config_kwargs(
    TIER, SPEC, output_dir=str(ROOT / "adapters" / SPEC.key),
    num_train_epochs=EPOCHS, mask_mode=MASK_MODE,
    total_steps=PLANNED_STEPS,
)
sft_kwargs, dropped = train.filter_kwargs(SFTConfig, want_sft, label="SFTConfig")
if dropped:
    print("⚠ TRL không nhận:", dropped)

want_lora = train.lora_config_kwargs(SPEC, targets)
lora_kwargs, _ = train.filter_kwargs(LoraConfig, want_lora, label="LoraConfig")

print(json.dumps({k: str(v) for k, v in sft_kwargs.items()}, indent=2)[:900])

# %% [markdown]
# ## 5. Train

# %%
generate.free_memory()
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(**sft_kwargs),
    train_dataset=train_ds,
    processing_class=tok,          # NOT tokenizer= — removed in TRL v1
    peft_config=LoraConfig(**lora_kwargs),
)

t0 = time.perf_counter()
result = trainer.train()
elapsed = time.perf_counter() - t0
print(f"train {elapsed:.0f}s  final loss {result.training_loss:.4f}")

# %% [markdown]
# ## 6. LƯU ADAPTER NGAY
#
# Trước khi eval. Eval có thể OOM; adapter thì đã an toàn trên đĩa.

# %%
out = ROOT / "adapters" / SPEC.key
trainer.model.save_pretrained(out)
tok.save_pretrained(out)
print("saved ->", out)

row = train.summarize_run(SPEC, TIER, targets, trainable, elapsed, generate.peak_vram_gb())
row["final_loss"] = round(result.training_loss, 4)
row["mask_mode"] = MASK_MODE
row["max_steps"] = PLANNED_STEPS
row["epochs"] = EPOCHS
report.append_row(row, results_dir=ROOT / "results")
print(json.dumps(row, ensure_ascii=False, indent=2))

# %% [markdown]
# ## ✅ Checkpoint NB3
# - [ ] `adapters/correct/` tồn tại
# - [ ] `results/runs.csv` có một dòng `correct`
# - [ ] Bạn đã ghi lại loss cuối và peak VRAM
