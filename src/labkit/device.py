"""Device and precision selection.

**Why this file exists.** The lab's default tier is a free-Colab **T4**, and a T4 is
Turing (compute capability 7.5). **Turing has no bfloat16 support** — bf16 needs
Ampere (8.0) or newer. Hardcoding `bf16=True`, which every 2026 tutorial does because
every 2026 tutorial is written on an A100, is wrong on the exact hardware this lab
tells students to use.

fp16 is the correct choice there, and it is not merely a flag swap: fp16 has a much
smaller exponent range, so training needs **gradient scaling** to avoid underflow.
Trainers enable a GradScaler automatically when told `fp16=True`, which is why the
precision decision must reach the training arguments rather than only the model load.
"""
from __future__ import annotations


def describe() -> dict:
    """What we are actually running on. Print this before training."""
    info = {"device": "cpu", "name": "CPU", "bf16": False, "fp16": False,
            "capability": None, "vram_gb": None}
    try:
        import torch
    except (ImportError, OSError, RuntimeError):         # pragma: no cover
        return info

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        # DO NOT use torch.cuda.is_bf16_supported() bare. Its signature is
        # `is_bf16_supported(including_emulation: bool = True)`, and with the default it
        # returns True on a T4 — it merely checks that a bf16 tensor can be *created*,
        # which Turing can do by emulation. Training then "works" at a large speed
        # penalty while reporting bf16=True. Hardware support is compute capability
        # >= 8.0 (Ampere), full stop.
        native_bf16 = major >= 8
        try:                       # belt and braces on builds that expose the kwarg
            native_bf16 = native_bf16 and torch.cuda.is_bf16_supported(
                including_emulation=False)
        except TypeError:          # older torch without the kwarg
            pass
        info.update(
            device="cuda",
            name=torch.cuda.get_device_name(0),
            capability=f"{major}.{minor}",
            vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
            bf16=bool(native_bf16),
            fp16=True,
        )
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS: fp16 works; bf16 support is patchy across torch versions, so do not
        # claim it. Training here is for smoke-testing the code path, not for results.
        info.update(device="mps", name="Apple MPS", fp16=True, bf16=False)
    return info


def precision(explicit: str | None = None) -> str:
    """Return 'bf16' | 'fp16' | 'fp32' for the current device."""
    if explicit:
        if explicit not in ("bf16", "fp16", "fp32"):
            raise ValueError(f"precision must be bf16|fp16|fp32, got {explicit!r}")
        return explicit
    info = describe()
    if info["bf16"]:
        return "bf16"
    if info["fp16"]:
        return "fp16"
    return "fp32"


def torch_dtype(explicit: str | None = None):
    """The torch dtype matching `precision()`."""
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        precision(explicit)
    ]


def banner() -> str:
    info = describe()
    p = precision()
    line = f"{info['name']} ({info['device']}"
    if info["capability"]:
        line += f", sm_{info['capability'].replace('.', '')}"
    if info["vram_gb"]:
        line += f", {info['vram_gb']} GB"
    line += f") -> precision={p}"
    if info["device"] == "cuda" and not info["bf16"]:
        line += ("\n  note: this GPU predates Ampere, so it has NO bfloat16. Using fp16 "
                 "with gradient scaling instead. Tutorials that hardcode bf16=True fail here.")
    return line

def has_flash_attention() -> bool:
    """Is a FlashAttention kernel importable?

    TRL warns that `padding_free=True` is only known to behave with
    flash_attention_2/3 kernels. FlashAttention-2 needs **Ampere (sm_80) or newer**, so
    on the lab's default T4 it is unavailable no matter what you install.
    """
    for mod in ("flash_attn", "kernels"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def supports_padding_free(min_batch: int = 2) -> bool:
    """Whether `padding_free=True` is both SAFE and USEFUL here.

    Two independent conditions, and the lab's default tier fails both:

    * **Safe** — needs a FlashAttention kernel. Padding-free flattens a batch into one
      sequence; without a kernel that understands the boundaries, attention can run
      across them. That is exactly deck §17.3's warning ("packing is free only when
      sequence boundaries are respected") applied to its sibling flag.
    * **Useful** — pointless at `per_device_train_batch_size=1`: there is no padding
      between sequences to remove when a batch holds one sequence. TRL says so
      outright: *"Using a batch size of 1 annihilates the benefits of padding-free."*
    """
    return has_flash_attention() and min_batch >= 2
