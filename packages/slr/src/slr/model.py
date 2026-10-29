from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Preset:
    timm_name: str
    family: str
    tier: str
    img_size: int = 224
    batch_size: int = 64
    lr: float = 3e-4
    layer_decay: float | None = None
    approx_params_m: float = 0.0


def _p(timm_name, family, tier, **kw) -> Preset:
    return Preset(timm_name=timm_name, family=family, tier=tier, **kw)


PRESETS: dict[str, Preset] = {
    "vit_tiny": _p("vit_tiny_patch16_224.augreg_in21k_ft_in1k", "vit", "scale",
                   batch_size=128, lr=5e-4, approx_params_m=5.5),
    "vit_small": _p("vit_small_patch16_224.augreg_in21k_ft_in1k", "vit", "scale",
                    batch_size=96, lr=3e-4, approx_params_m=21.7),
    "vit_base": _p("vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit", "scale",
                   batch_size=48, lr=1e-4, layer_decay=0.75, approx_params_m=85.8),
    "vit_large": _p("vit_large_patch16_224.augreg_in21k_ft_in1k", "vit", "scale",
                    batch_size=16, lr=5e-5, layer_decay=0.70, approx_params_m=303.3),

    # Same architecture as vit_base with a better training recipe.
    "deit3_base": _p("deit3_base_patch16_224.fb_in22k_ft_in1k", "vit", "arch",
                     batch_size=48, lr=1e-4, layer_decay=0.75, approx_params_m=85.8),
    "swin_base": _p("swin_base_patch4_window7_224.ms_in22k_ft_in1k", "hybrid", "arch",
                    batch_size=48, lr=1e-4, approx_params_m=86.8),
    # CNN baselines to compare the transformers against.
    "convnext_base": _p("convnext_base.fb_in22k_ft_in1k", "cnn", "arch",
                        batch_size=48, lr=1e-4, approx_params_m=87.6),
    "convnextv2_base": _p("convnextv2_base.fcmae_ft_in22k_in1k", "cnn", "arch",
                          batch_size=48, lr=1e-4, approx_params_m=87.7),
    "beitv2_base": _p("beitv2_base_patch16_224.in1k_ft_in22k_in1k", "vit", "arch",
                      batch_size=48, lr=1e-4, layer_decay=0.75, approx_params_m=85.8),
    "caformer_b36": _p("caformer_b36.sail_in22k_ft_in1k", "hybrid", "arch",
                       batch_size=32, lr=1e-4, approx_params_m=95.8),
    # 448px, about 4x the compute of the others.
    "eva02_base": _p("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", "vit", "arch",
                     img_size=448, batch_size=8, lr=5e-5, layer_decay=0.75,
                     approx_params_m=86.4),
    "siglip_base": _p("vit_base_patch16_siglip_224.webli", "vit", "arch",
                      batch_size=48, lr=1e-4, layer_decay=0.75, approx_params_m=92.9),
    "clip_base": _p("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", "vit", "arch",
                    batch_size=48, lr=1e-4, layer_decay=0.75, approx_params_m=85.8),
    "dinov2_small": _p("vit_small_patch14_dinov2.lvd142m", "vit", "arch",
                       img_size=518, batch_size=16, lr=1e-4, approx_params_m=22.1),
    "dinov2_base": _p("vit_base_patch14_reg4_dinov2.lvd142m", "vit", "arch",
                      img_size=518, batch_size=8, lr=5e-5, layer_decay=0.75,
                      approx_params_m=86.6),

    "efficientformer_s2": _p("efficientformerv2_s2.snap_dist_in1k", "hybrid", "mobile",
                             batch_size=128, lr=5e-4, approx_params_m=12.1),
    "mobilevit_v2": _p("mobilevitv2_150.cvnets_in22k_ft_in1k", "hybrid", "mobile",
                       img_size=256, batch_size=96, lr=5e-4, approx_params_m=9.8),
    "effnetv2_s": _p("tf_efficientnetv2_s.in21k_ft_in1k", "cnn", "mobile",
                     batch_size=64, lr=5e-4, approx_params_m=20.2),
}

SWEEPS: dict[str, list[str]] = {
    "scale": [k for k, v in PRESETS.items() if v.tier == "scale"],
    "arch": [k for k, v in PRESETS.items() if v.tier == "arch"],
    "mobile": [k for k, v in PRESETS.items() if v.tier == "mobile"],
    "quick": ["vit_small", "convnext_base", "siglip_base", "dinov2_small"],
    "all": list(PRESETS),
}


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build(preset: str, n_classes: int, drop_path: float = 0.1,
          pretrained: bool = True) -> nn.Module:
    import timm

    m = timm.create_model(PRESETS[preset].timm_name, pretrained=pretrained,
                          num_classes=n_classes, drop_path_rate=drop_path)
    m.slr_preset = preset
    return m


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def param_groups(m: nn.Module, lr: float, weight_decay: float = 0.05,
                 layer_decay: float | None = None) -> list[dict]:
    """Optimiser groups: no weight decay on norms and biases, optional layer-wise LR decay."""
    n_layers = len(getattr(m, "blocks", []) or [])
    if layer_decay is not None and not n_layers:
        print(f"[model] {type(m).__name__} has no flat .blocks; "
              "ignoring layer_decay and using a uniform learning rate")
        layer_decay = None

    def _decay(name: str) -> float:
        if layer_decay is None:
            return 1.0
        if name.startswith(("cls_token", "pos_embed", "patch_embed")):
            depth = 0
        elif name.startswith("blocks."):
            depth = int(name.split(".")[1]) + 1
        else:
            depth = n_layers + 1
        return layer_decay ** (n_layers + 1 - depth)

    groups: dict[tuple[float, bool], dict] = {}
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = p.ndim <= 1 or name.endswith(".bias")
        key = (_decay(name), no_wd)
        g = groups.setdefault(key, {
            "params": [], "lr": lr * key[0],
            "weight_decay": 0.0 if no_wd else weight_decay,
        })
        g["params"].append(p)
    return list(groups.values())


@torch.enable_grad()
def grad_attention(model: nn.Module, x: torch.Tensor, class_idx: int | None = None
                   ) -> torch.Tensor:
    """Chefer et al. gradient-weighted attention rollout: a (B, H, W) map."""
    blocks = getattr(model, "blocks", None)
    if not blocks:
        raise TypeError(f"{type(model).__name__} has no .blocks; not a timm ViT")

    maps: list[torch.Tensor] = []
    handles, saved_fused = [], []

    def keep(_m, inp, _out):
        att = inp[0]
        att.retain_grad()
        maps.append(att)

    for blk in blocks:
        saved_fused.append(getattr(blk.attn, "fused_attn", None))
        blk.attn.fused_attn = False
        handles.append(blk.attn.attn_drop.register_forward_hook(keep))
    try:
        out = model(x)
        idx = int(out[0].argmax()) if class_idx is None else class_idx
        model.zero_grad(set_to_none=True)
        out[0, idx].backward()
    finally:
        for h in handles:
            h.remove()
        for blk, f in zip(blocks, saved_fused):
            if f is not None:
                blk.attn.fused_attn = f

    n = maps[0].shape[-1]
    eye = torch.eye(n, device=x.device)
    roll = eye.clone()
    for att in maps:
        if att.grad is None:
            continue
        w = (att.grad * att).clamp(min=0).mean(dim=1)[0] + eye
        roll = (w / w.sum(dim=-1, keepdim=True)) @ roll

    mask = roll[0, 1:].detach()
    side = int(mask.numel() ** 0.5)
    mask = mask[: side * side].reshape(1, side, side)
    mask = mask - mask.amin()
    return mask / mask.amax().clamp_min(1e-8)
