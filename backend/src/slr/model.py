"""Backbones, the architecture search, and attention rollout.

The zoo is a dict, not a framework. Two axes are searched:

  * **scale** - ViT from 5.7M to 303M parameters, to find where the corpora stop
    rewarding capacity. On honest splits that happens far earlier than the
    leaderboards suggest.
  * **architecture** - a dozen backbones held at roughly base scale, so the
    comparison is about inductive bias and pretraining rather than parameter
    count.

`convnext_base` and `convnextv2_base` are in the zoo as controls, not
also-rans. If a plain CNN matches the transformers on the cross-corpus test,
then attention was not what mattered here and the write-up has to say so.

The language-supervised (`siglip`, `clip`) and self-supervised (`dinov2`,
`beitv2`, `eva02`) entries are the ones to watch. Their pretraining saw far more
visual variety than ImageNet, which is exactly the axis the cross-corpus test
stresses, so they are the most likely to hold up when the signer, room, and
camera all change at once.

Selection discipline: the winner is chosen on **validation** macro-F1 and only
then evaluated on test, once. Picking the architecture by its test score would
reintroduce, at the level of the experiment, precisely the leakage this project
exists to eliminate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Preset:
    timm_name: str
    family: str                    # "vit" | "hybrid" | "cnn"
    tier: str                      # "scale" | "arch" | "mobile"
    img_size: int = 224
    batch_size: int = 64           # sized for a 16 GB T4; halve on smaller cards
    lr: float = 3e-4
    layer_decay: float | None = None
    approx_params_m: float = 0.0   # measured with a 29-class head
    note: str = ""


def _p(timm_name, family, tier, **kw) -> Preset:
    return Preset(timm_name=timm_name, family=family, tier=tier, **kw)


PRESETS: dict[str, Preset] = {
    # --- scale ladder: same architecture, four capacities ---------------------
    "vit_tiny": _p("vit_tiny_patch16_224.augreg_in21k_ft_in1k", "vit", "scale",
                   batch_size=128, lr=5e-4, approx_params_m=5.5,
                   note="the honest-accuracy floor, and fast enough to iterate on"),
    "vit_small": _p("vit_small_patch16_224.augreg_in21k_ft_in1k", "vit", "scale",
                    batch_size=96, lr=3e-4, approx_params_m=21.7,
                    note="best accuracy per FLOP in the ViT family"),
    "vit_base": _p("vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit", "scale",
                   batch_size=48, lr=1e-4, layer_decay=0.75, approx_params_m=85.8,
                   note="the standard research comparison point"),
    "vit_large": _p("vit_large_patch16_224.augreg_in21k_ft_in1k", "vit", "scale",
                    batch_size=16, lr=5e-5, layer_decay=0.70, approx_params_m=303.3,
                    note="A100-class. Included to show where returns stop, not "
                         "because it is expected to win"),

    # --- architecture comparison, held near base scale ------------------------
    "deit3_base": _p("deit3_base_patch16_224.fb_in22k_ft_in1k", "vit", "arch",
                     batch_size=48, lr=1e-4, layer_decay=0.75, approx_params_m=85.8,
                     note="same architecture as vit_base, better training recipe. "
                          "Isolates recipe from architecture"),
    "swin_base": _p("swin_base_patch4_window7_224.ms_in22k_ft_in1k", "hybrid", "arch",
                    batch_size=48, lr=1e-4, approx_params_m=86.8,
                    note="hierarchical windowed attention; a locality prior that "
                         "plain ViT lacks and hands may reward"),
    "convnext_base": _p("convnext_base.fb_in22k_ft_in1k", "cnn", "arch",
                        batch_size=48, lr=1e-4, approx_params_m=87.6,
                        note="CONTROL. A pure CNN on a modern recipe. If this ties "
                             "the transformers, attention was not the story"),
    "convnextv2_base": _p("convnextv2_base.fcmae_ft_in22k_in1k", "cnn", "arch",
                          batch_size=48, lr=1e-4, approx_params_m=87.7,
                          note="CNN control with masked-autoencoder pretraining"),
    "beitv2_base": _p("beitv2_base_patch16_224.in1k_ft_in22k_in1k", "vit", "arch",
                      batch_size=48, lr=1e-4, layer_decay=0.75, approx_params_m=85.8,
                      note="masked image modelling; strong on fine-grained texture"),
    "caformer_b36": _p("caformer_b36.sail_in22k_ft_in1k", "hybrid", "arch",
                       batch_size=32, lr=1e-4, approx_params_m=95.8,
                       note="MetaFormer, conv stages then attention stages. Among "
                            "the strongest ImageNet models at this size"),
    "eva02_base": _p("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", "vit", "arch",
                     img_size=448, batch_size=8, lr=5e-5, layer_decay=0.75,
                     approx_params_m=86.4,
                     note="highest ImageNet accuracy in the zoo, at 448px and roughly "
                          "4x the compute. The SOTA ceiling check"),
    "siglip_base": _p("vit_base_patch16_siglip_224.webli", "vit", "arch",
                      batch_size=48, lr=1e-4, layer_decay=0.75, approx_params_m=92.9,
                      note="language-supervised on web images. Pretrained on far more "
                           "visual variety than ImageNet, so a strong candidate for "
                           "the cross-corpus test specifically"),
    "clip_base": _p("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", "vit", "arch",
                    batch_size=48, lr=1e-4, layer_decay=0.75, approx_params_m=85.8,
                    note="LAION-2B pretraining; the other language-supervised entry"),
    "dinov2_small": _p("vit_small_patch14_dinov2.lvd142m", "vit", "arch",
                       img_size=518, batch_size=16, lr=1e-4, approx_params_m=22.1,
                       note="self-supervised features at small scale"),
    "dinov2_base": _p("vit_base_patch14_reg4_dinov2.lvd142m", "vit", "arch",
                      img_size=518, batch_size=8, lr=5e-5, layer_decay=0.75,
                      approx_params_m=86.6,
                      note="DINOv2 with registers. Self-supervised features are the "
                           "usual winner on out-of-distribution transfer"),

    # --- mobile tier: what could eventually run on the phone itself -----------
    "efficientformer_s2": _p("efficientformerv2_s2.snap_dist_in1k", "hybrid", "mobile",
                             batch_size=128, lr=5e-4, approx_params_m=12.1,
                             note="designed for phone latency; the on-device candidate"),
    "mobilevit_v2": _p("mobilevitv2_150.cvnets_in22k_ft_in1k", "hybrid", "mobile",
                       img_size=256, batch_size=96, lr=5e-4, approx_params_m=9.8,
                       note="transformer blocks at mobile cost"),
    "effnetv2_s": _p("tf_efficientnetv2_s.in21k_ft_in1k", "cnn", "mobile",
                     batch_size=64, lr=5e-4, approx_params_m=20.2,
                     note="CNN mobile control"),
}

# Named sweeps. `python -m slr.train manifest.csv --preset scale` runs one.
SWEEPS: dict[str, list[str]] = {
    "scale": [k for k, v in PRESETS.items() if v.tier == "scale"],
    "arch": [k for k, v in PRESETS.items() if v.tier == "arch"],
    "mobile": [k for k, v in PRESETS.items() if v.tier == "mobile"],
    # A single T4 session that still answers the question: one from each family
    # plus the CNN control, at comparable cost.
    "quick": ["vit_small", "convnext_base", "siglip_base", "dinov2_small"],
    "all": list(PRESETS),
}
SWEEP = SWEEPS["scale"]  # backwards-compatible default


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build(preset: str, n_classes: int, drop_path: float = 0.1,
          pretrained: bool = True) -> nn.Module:
    import timm

    p = PRESETS[preset]
    kw = {"pretrained": pretrained, "num_classes": n_classes}
    try:
        m = timm.create_model(p.timm_name, drop_path_rate=drop_path, **kw)
    except TypeError:                     # a few backbones reject drop_path_rate
        m = timm.create_model(p.timm_name, **kw)
    m.slr_preset = preset  # type: ignore[attr-defined]
    return m


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def param_groups(m: nn.Module, lr: float, weight_decay: float = 0.05,
                 layer_decay: float | None = None) -> list[dict]:
    """Optimiser groups with no weight decay on norms/bias, plus optional
    layer-wise LR decay.

    Layer-wise decay matters here: the corpora are small and visually narrow,
    so a uniform LR drags the pretrained early blocks toward the training
    corpus's background statistics, which is precisely the overfitting the
    cross-corpus test is designed to expose.
    """
    n_layers = len(getattr(m, "blocks", []) or [])
    if layer_decay is not None and not n_layers:
        # Only the flat ViT-style `.blocks` stack has an unambiguous depth index.
        # Hierarchical backbones (Swin, ConvNeXt, CAFormer) would need a per-family
        # depth map, which is not worth it - they train fine on a uniform LR.
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


# --- explainability ----------------------------------------------------------

@torch.no_grad()
def attention_rollout(model: nn.Module, x: torch.Tensor,
                      head_fusion: str = "mean", discard_ratio: float = 0.9
                      ) -> torch.Tensor:
    """Abnar & Zuidema attention rollout: a (B, H, W) map over the input.

    Multiplies the per-block attention matrices (plus identity, for the residual
    stream) to trace how much each patch contributes to the class token. Serves
    two purposes: it is the evidence image the phone app can show next to a
    prediction, and it is the diagnostic that catches a model attending to the
    wall behind the signer instead of the hand.
    """
    blocks = getattr(model, "blocks", None)
    if not blocks:
        raise TypeError(f"{type(model).__name__} has no .blocks; not a timm ViT")

    maps: list[torch.Tensor] = []
    handles = []
    saved_fused = []
    for blk in blocks:
        attn = blk.attn
        saved_fused.append(getattr(attn, "fused_attn", None))
        attn.fused_attn = False  # fused SDPA never materialises the matrix
        handles.append(attn.attn_drop.register_forward_hook(
            lambda _m, inp, _out: maps.append(inp[0].detach())))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
        for blk, f in zip(blocks, saved_fused):
            if f is not None:
                blk.attn.fused_attn = f

    b, _, n, _ = maps[0].shape
    roll = torch.eye(n, device=x.device).expand(b, n, n).clone()
    for a in maps:
        a = a.max(dim=1).values if head_fusion == "max" else a.mean(dim=1)
        if discard_ratio > 0:  # drop the long tail of weak, noisy links
            k = int(a.shape[-1] * discard_ratio)
            if k > 0:
                thresh = a.kthvalue(k, dim=-1, keepdim=True).values
                a = a.masked_fill(a < thresh, 0.0)
        a = a + torch.eye(n, device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        roll = a @ roll

    n_prefix = n - (n - 1)  # cls token(s); timm ViTs use exactly one here
    mask = roll[:, 0, n_prefix:]
    side = int(mask.shape[-1] ** 0.5)
    mask = mask[:, : side * side].reshape(b, side, side)
    mask = mask - mask.amin(dim=(1, 2), keepdim=True)
    return mask / mask.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)


@torch.enable_grad()
def grad_attention(model: nn.Module, x: torch.Tensor, class_idx: int | None = None
                   ) -> torch.Tensor:
    """Chefer et al. gradient-weighted attention rollout: a (B, H, W) map.

    Plain `attention_rollout` above is **class-agnostic** - it multiplies the
    attention matrices without ever consulting the prediction, so it returns the
    same picture whether the model said B or T, and it cannot answer "why this
    letter". Worse, it faithfully reports the attention paid to the handful of
    low-information background patches that ViTs repurpose as global scratchpads
    (the "register" phenomenon), which is why its mass lands in the corners.

    Measured on this checkpoint against occlusion ground truth - blank each
    16x16 patch, record the drop in the predicted class - over four cross-corpus
    B images:

        rollout (discard 0.9)     -0.04  -0.07  -0.13  -0.05
        grad-weighted rollout     +0.32  +0.11  +0.22  +0.11

    Weighting each attention matrix by its own gradient before rolling up keeps
    only the heads that actually moved the predicted logit, which is what makes
    the map class-specific. It is an improvement, not a solution: +0.1 to +0.3
    is a weak correlation, and any caption shown next to it should say so.
    """
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
        # Negative contributions are dropped: they are evidence against the
        # class, and averaging them in cancels out the signal we want to show.
        w = (att.grad * att).clamp(min=0).mean(dim=1)[0] + eye
        roll = (w / w.sum(dim=-1, keepdim=True)) @ roll

    mask = roll[0, 1:].detach()
    side = int(mask.numel() ** 0.5)
    mask = mask[: side * side].reshape(1, side, side)
    mask = mask - mask.amin()
    return mask / mask.amax().clamp_min(1e-8)
