"""Placement resolution, parameter accounting, and version-defensive config building."""
import dataclasses
import warnings

import pytest
from labkit import modeling, train
from labkit.config import CONTRAST_EPOCHS, SPECS, TIERS, get_tier


class _Lin:
    def __init__(self, i, o):
        self.in_features, self.out_features = i, o


class _Model:
    """Qwen3.5-4B-shaped: 32 text layers + a 24-block vision tower + lm_head."""

    def __init__(self, vision_uses_same_names=False):
        H, Q, KV, I = 2560, 4096, 1024, 9728
        self.mods = []
        for i in range(32):
            for n, (a, b) in {
                "q_proj": (H, Q), "k_proj": (H, KV), "v_proj": (H, KV), "o_proj": (Q, H),
                "gate_proj": (H, I), "up_proj": (H, I), "down_proj": (I, H),
            }.items():
                self.mods.append((f"model.language_model.layers.{i}.{n}", _Lin(a, b)))
        vname = "q_proj" if vision_uses_same_names else "qkv"
        for i in range(24):
            self.mods.append((f"model.visual.blocks.{i}.{vname}", _Lin(1024, 3072)))
        self.mods.append(("lm_head", _Lin(H, 248320)))

    def named_modules(self):
        return iter(self.mods)


def test_vision_tower_is_excluded():
    m = _Model()
    targets = modeling.resolve_target_modules(m, "text-linear")
    assert "qkv" not in targets
    assert set(targets) == {"q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"}


def test_lm_head_and_embeddings_excluded():
    assert "lm_head" not in modeling.resolve_target_modules(_Model(), "text-linear")


def test_suffix_collision_falls_back_to_full_names():
    """If the vision tower also calls a layer `q_proj`, suffix targeting would hit it."""
    m = _Model(vision_uses_same_names=True)
    targets = modeling.resolve_target_modules(m, "text-linear")
    assert all("." in t for t in targets), "expected fully-qualified names on collision"
    assert not any("visual" in t for t in targets)


def test_attn_only_is_just_q_and_v():
    assert modeling.resolve_target_modules(_Model(), "attn-only") == ["q_proj", "v_proj"]


def test_attn_full_includes_k_and_o():
    assert modeling.resolve_target_modules(_Model(), "attn-full") == \
        ["k_proj", "o_proj", "q_proj", "v_proj"]


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        modeling.resolve_target_modules(_Model(), "everything-everywhere")


def test_lora_params_scale_linearly_with_rank():
    m = _Model()
    t = modeling.resolve_target_modules(m, "text-linear")
    assert modeling.count_lora_params(m, t, 32) == 2 * modeling.count_lora_params(m, t, 16)


def test_matched_rank_equalizes_the_parameter_budget():
    """The fair-contrast guarantee: same budget, only placement differs."""
    m = _Model()
    text = modeling.resolve_target_modules(m, "text-linear")
    qv = modeling.resolve_target_modules(m, "attn-only")
    r = modeling.matched_rank(m, text, 16, qv)
    budget = modeling.count_lora_params(m, text, 16)
    matched = modeling.count_lora_params(m, qv, r)
    assert abs(matched - budget) / budget < 0.02, "budgets must match within 2%"
    assert r > 16, "attention-only needs a HIGHER rank to reach the same budget"


def test_describe_placement_shows_the_naive_comparison_is_unfair():
    rows = {r["placement"]: r for r in modeling.describe_placement(_Model(), 16)}
    naive = rows["attn-only(q,v)"]["trainable"]
    full = rows["text-linear"]["trainable"]
    assert naive < full / 2, "same-rank q,v has far fewer params — comparing them proves nothing"


def test_layer_type_summary_surfaces_hybrid_attention():
    class Cfg:
        num_hidden_layers = 32
        full_attention_interval = 4
        linear_num_key_heads = 16
        layer_types = ["linear_attention"] * 24 + ["full_attention"] * 8
    s = modeling.layer_type_summary(Cfg())
    assert s["layer_types"] == {"linear_attention": 24, "full_attention": 8}
    assert s["full_attention_interval"] == 4


# --- version-defensive config ----------------------------------------------

@dataclasses.dataclass
class _OldSFTConfig:
    output_dir: str = "."
    max_length: int = 512
    learning_rate: float = 1e-4
    packing: bool = False
    # deliberately missing: padding_free, loss_type, assistant_only_loss


def test_filter_kwargs_drops_unknown_and_warns():
    desired = train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "out")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kept, dropped = train.filter_kwargs(_OldSFTConfig, desired, label="SFTConfig")
    assert "padding_free" in dropped and "loss_type" in dropped
    assert "max_length" in kept and "learning_rate" in kept
    assert caught and "does not accept" in str(caught[0].message)
    _OldSFTConfig(**kept)          # must actually construct


def test_sft_kwargs_use_the_post_v1_names():
    kw = train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "out")
    assert "max_length" in kw and "max_seq_length" not in kw
    assert "eval_strategy" not in kw or "evaluation_strategy" not in kw
    assert kw["loss_type"] == "chunked_nll"


def test_assistant_only_loss_is_never_set():
    """Regression (F-10): TRL's assistant_only_loss silently supervises ZERO tokens on
    templates without {% generation %} markers, which includes Qwen3.5. The lab
    pre-tokenizes with its own verified mask instead."""
    kw = train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "out")
    assert "assistant_only_loss" not in kw
    assert kw["packing"] is False, "pre-tokenized labels are incompatible with packing"


# --- precision must follow the device, not the fashion -----------------------
# The lab's default tier is a free-Colab T4 = Turing = NO bfloat16. Hardcoding
# bf16=True (as most 2026 tutorials do, because they were written on A100s) breaks
# on the exact hardware this lab recommends.

@pytest.mark.parametrize("prec,expect_bf16,expect_fp16", [
    ("bf16", True, False),
    ("fp16", False, True),
    ("fp32", False, False),
])
def test_precision_flags_are_mutually_exclusive(prec, expect_bf16, expect_fp16):
    kw = train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "o", precision=prec)
    assert kw["bf16"] is expect_bf16
    assert kw["fp16"] is expect_fp16
    assert not (kw["bf16"] and kw["fp16"]), "never set both"


def test_precision_is_never_hardcoded():
    """Regression: an earlier version always emitted bf16=True."""
    from labkit import device
    kw = train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "o")
    assert kw["bf16"] is (device.precision() == "bf16")


def test_bad_precision_rejected():
    from labkit import device
    with pytest.raises(ValueError, match="bf16"):
        device.precision("float8")


def test_describe_reports_a_device():
    from labkit import device
    info = device.describe()
    assert info["device"] in ("cpu", "cuda", "mps")
    assert device.precision() in ("bf16", "fp16", "fp32")


def test_banner_warns_when_gpu_lacks_bf16(monkeypatch):
    """A T4-shaped device must produce the explanatory note."""
    from labkit import device
    monkeypatch.setattr(device, "describe", lambda: {
        "device": "cuda", "name": "Tesla T4", "bf16": False, "fp16": True,
        "capability": "7.5", "vram_gb": 14.6})
    text = device.banner()
    assert "precision=fp16" in text and "NO bfloat16" in text and "sm_75" in text


def test_effective_batch_ceiling_is_enforced():
    from labkit.config import Tier
    bad = Tier("BAD", "m", 1, 1024, 8, 8, "")     # 8*8 = 64 > 32
    with pytest.raises(ValueError, match="effective batch"):
        train.sft_config_kwargs(bad, SPECS["correct"], "out")


def test_every_shipped_tier_respects_the_ceiling():
    for name, tier in TIERS.items():
        assert tier.effective_batch <= 32, f"tier {name} violates the §10.4 batch rule"


def test_lora_kwargs_reject_an_unresolved_rank():
    with pytest.raises(ValueError, match="unresolved rank"):
        train.lora_config_kwargs(SPECS["attn_only"], ["q_proj"])


def test_lora_kwargs_keep_the_alpha_equals_2r_invariant():
    kw = train.lora_config_kwargs(SPECS["attn_only"].resolved(90), ["q_proj", "v_proj"])
    assert kw["r"] == 90 and kw["lora_alpha"] == 180


def test_mask_mode_does_not_leak_into_trl_flags():
    """mask_mode is applied when pre-tokenizing, not via a TRL flag."""
    for mode in ("assistant-only", "everything", "response-only"):
        kw = train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "o", mask_mode=mode)
        assert "assistant_only_loss" not in kw and "completion_only_loss" not in kw



# --- padding_free must be conditional (F-14) --------------------------------

def test_padding_free_off_without_flash_attention(monkeypatch):
    """T4 case: no FlashAttention -> padding_free must be off, and max_length kept."""
    from labkit import device
    monkeypatch.setattr(device, "has_flash_attention", lambda: False)
    kw = train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "o")
    assert kw["padding_free"] is False
    assert kw["max_length"] == get_tier("T4").max_length


def test_padding_free_off_at_batch_size_one(monkeypatch):
    """Even WITH flash-attn, batch=1 makes padding-free pointless."""
    from labkit import device
    monkeypatch.setattr(device, "has_flash_attention", lambda: True)
    assert get_tier("T4").per_device_batch == 1
    assert train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "o")["padding_free"] is False


def test_padding_free_on_sets_max_length_none(monkeypatch):
    """TRL rejects padding_free + max_length without packing; we pre-truncate, so None."""
    from labkit import device
    monkeypatch.setattr(device, "has_flash_attention", lambda: True)
    kw = train.sft_config_kwargs(get_tier("BIGGPU"), SPECS["correct"], "o")
    assert get_tier("BIGGPU").per_device_batch >= 2
    assert kw["padding_free"] is True and kw["max_length"] is None


def test_emulated_bf16_does_not_count_as_bf16_support(monkeypatch):
    """F-15: torch.cuda.is_bf16_supported() defaults to including_emulation=True and
    returns True on a T4. Capability >= 8.0 is the real test.

    Skipped without torch: `requirements-cpu.txt` ships no torch by design, and this is
    the only test in the suite that needs it. An unguarded import here made
    `make setup-cpu && make smoke` report "Not ready to submit" for every student
    working the GPU-less slice the README recommends starting from.
    """
    torch = pytest.importorskip("torch")
    from labkit import device

    class _Props:
        total_memory = 15 * 1024 ** 3

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability(_):
            return (7, 5)                      # Turing / T4

        @staticmethod
        def get_device_name(_):
            return "Tesla T4"

        @staticmethod
        def get_device_properties(_):
            return _Props()

        @staticmethod
        def is_bf16_supported(including_emulation=True):
            return True                        # what torch actually does on a T4

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    info = device.describe()
    assert info["capability"] == "7.5"
    assert info["bf16"] is False, "emulated bf16 must not be reported as support"
    assert device.precision() == "fp16"


# --- warmup + step budget (F-16) --------------------------------------------
# transformers v5 / TRL 1.10 deleted `warmup_ratio`; only `warmup_steps` survives.
# Passing the ratio does not raise -- filter_kwargs drops it with a warning -- so a run
# that "looks fine" trains with no warmup at all. Measured on Colab 2026-08-20:
# [f.name for f in dataclasses.fields(SFTConfig) if "warm" in f.name] == ["warmup_steps"]

def test_warmup_ratio_is_never_emitted():
    for kw in (train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "o"),
               train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "o", max_steps=30),
               train.sft_config_kwargs(get_tier("T4"), SPECS["correct"], "o", total_steps=30)):
        assert "warmup_ratio" not in kw, "removed in transformers v5 — would be silently dropped"


def test_warmup_steps_follow_the_step_budget():
    assert train.sft_config_kwargs(
        get_tier("T4"), SPECS["correct"], "o", total_steps=30)["warmup_steps"] == 3
    assert train.sft_config_kwargs(
        get_tier("T4"), SPECS["correct"], "o", max_steps=30)["warmup_steps"] == 3
    # never zero: round(0.1 * 4) == 0 would mean "warmup requested, none applied"
    assert train.sft_config_kwargs(
        get_tier("T4"), SPECS["correct"], "o", max_steps=4)["warmup_steps"] == 1


def test_planned_steps_matches_the_measured_nb3_run():
    """225 train rows, T4 (batch 1 x grad_accum 16), 2 epochs -> the 30 steps the real
    T4 run printed as `100% 30/30`."""
    assert train.planned_steps(225, get_tier("T4"), 2.0) == 30


def test_contrasts_get_the_same_step_budget_as_the_baseline():
    """NB4 used to hardcode max_steps=60 while NB3 ran 30 — every contrast was trained
    twice as long as the baseline it is compared against."""
    tier = get_tier("T4")
    nb3 = train.planned_steps(225, tier, 2.0)
    nb4 = train.planned_steps(225, tier, CONTRAST_EPOCHS)
    assert nb3 == nb4 == 30
