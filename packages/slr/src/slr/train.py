"""Fine-tuning loop and the architecture sweep.

Model selection is on validation macro-F1 rather than accuracy: the corpora are
imbalanced and a few classes (`del`, `nothing`) are tiny, so plain accuracy can
be bought by ignoring them.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import data, hands, model as M, safe_load, sources


def _accumulation_group_size(i: int, total: int, accumulate: int) -> int:
    """Microbatches contributing to `i`'s optimizer step, including a short tail."""
    if accumulate < 1:
        raise ValueError("accumulate must be at least one")
    start = (i // accumulate) * accumulate
    return min(accumulate, total - start)


def _should_optimizer_step(i: int, total: int, accumulate: int) -> bool:
    return (i + 1) % accumulate == 0 or i + 1 == total


def _macro_f1(conf: torch.Tensor) -> float:
    tp = conf.diag().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    present = (conf.sum(1) > 0)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1e-9)
    return f1[present].mean().item() if present.any() else 0.0


@torch.no_grad()
def evaluate_split(net: nn.Module, loader, dev: torch.device, n_classes: int
                   ) -> tuple[float, float, float]:
    """Return (loss, accuracy, macro-F1) over one loader."""
    net.eval()
    crit = nn.CrossEntropyLoss()
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = net(x)
            loss = crit(out, y)
        total_loss += loss.item() * y.numel()
        n += y.numel()
        for t, p in zip(y.cpu(), out.argmax(1).cpu()):
            conf[t, p] += 1
    acc = conf.diag().sum().item() / max(n, 1)
    return total_loss / max(n, 1), acc, _macro_f1(conf)


def train_one(rows: list[dict], preset: str, epochs: int = 12, out_dir: Path = Path("runs"),
              workers: int = 4, label_smoothing: float = 0.1, seed: int = 0,
              tag: str = "", eval_test: bool = True,
              batch_size: int | None = None, accumulate: int = 1) -> dict:
    """Fine-tune one backbone.

    Use `eval_test=False` in a sweep so test is only scored for the winner.
    """
    torch.manual_seed(seed)
    p = M.PRESETS[preset]
    dev = M.device()
    n_classes = len(sources.CLASSES)

    micro_batch = batch_size or p.batch_size
    dl = data.loaders(rows, p.img_size, micro_batch, workers)
    if "train" not in dl:
        raise ValueError("no train rows in manifest")

    # Work out hand_crop from the image paths so the checkpoint can't disagree
    # with its data.
    n_crop = sum(str(hands.CROP_ROOT) in r["path"] for r in rows)
    if 0 < n_crop < len(rows):
        raise ValueError(
            f"only {n_crop} of {len(rows)} rows are hand crops; run "
            "`python -m slr.hands crop` over the whole manifest"
        )
    hand_crop = n_crop == len(rows)
    print(f"[train] hand-cropped input: {hand_crop}")

    net = M.build(preset, n_classes).to(dev)
    print(f"[train] {preset} ({M.n_params(net) / 1e6:.1f}M params) on {dev}, "
          f"{len(dl['train'].dataset)} train imgs, microbatch={micro_batch}, "
          f"accumulate={accumulate}, effective_batch={micro_batch * accumulate}")

    opt = torch.optim.AdamW(
        M.param_groups(net, p.lr, layer_decay=p.layer_decay), lr=p.lr, betas=(0.9, 0.999))
    steps = max(1, epochs * math.ceil(len(dl["train"]) / accumulate))
    warmup = min(500, steps // 10)

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        prog = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    run = out_dir / (tag or f"{preset}_{int(time.time())}")
    run.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = {"val_f1": -1.0}

    for ep in range(epochs):
        net.train()
        t0, seen, run_loss = time.time(), 0, 0.0
        total_micro = len(dl["train"])
        opt.zero_grad(set_to_none=True)
        for micro_i, (x, y) in enumerate(dl["train"]):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = crit(net(x), y)
            group_size = _accumulation_group_size(micro_i, total_micro, accumulate)
            scaler.scale(loss / group_size).backward()
            if _should_optimizer_step(micro_i, total_micro, accumulate):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            run_loss += loss.item() * y.numel()
            seen += y.numel()

        rec = {"epoch": ep, "train_loss": run_loss / max(seen, 1),
               "lr": sched.get_last_lr()[0], "secs": round(time.time() - t0, 1)}
        if "val" in dl:
            vl, va, vf = evaluate_split(net, dl["val"], dev, n_classes)
            rec |= {"val_loss": vl, "val_acc": va, "val_f1": vf}
            if vf > best["val_f1"]:
                best = {"epoch": ep, "val_f1": vf, "val_acc": va}
                torch.save({"preset": preset, "classes": sources.CLASSES,
                            "img_size": p.img_size, "hand_crop": hand_crop,
                            "state_dict": net.state_dict()},
                           run / "best.pt")
        history.append(rec)
        print(f"[train] ep{ep:02d} " + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))

    result = {"preset": preset, "params_m": round(M.n_params(net) / 1e6, 2),
              "epochs": epochs, "batch_size": micro_batch,
              "gradient_accumulation": accumulate,
              "effective_batch_size": micro_batch * accumulate,
              "best": best, "history": history}

    if eval_test and "test" in dl:
        if (run / "best.pt").exists():
            net.load_state_dict(safe_load(run / "best.pt", map_location=dev)["state_dict"])
        tl, ta, tf = evaluate_split(net, dl["test"], dev, n_classes)
        result["test"] = {"loss": tl, "acc": ta, "macro_f1": tf}
        print(f"[train] TEST acc={ta:.4f} macro_f1={tf:.4f}")

    (run / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[ok] {run}")
    return result


def sweep(rows: list[dict], presets: list[str], epochs: int = 8,
          out_dir: Path = Path("runs"), workers: int = 4, seed: int = 0,
          tag: str = "sweep") -> dict:
    """Train every preset, rank on validation, then evaluate only the winner on test."""
    if not any(r["split"] == "val" for r in rows):
        raise ValueError(
            "an architecture search needs a validation split to select on. "
            "Re-split with `python -m slr.data <manifest> --val 0.15`."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    board_path = out_dir / f"{tag}_leaderboard.json"

    # Saved after every model so an interrupted sweep can pick up where it stopped.
    board: list[dict] = []
    if board_path.exists():
        board = json.loads(board_path.read_text()).get("leaderboard", [])
        print(f"[sweep] resuming: {len(board)} candidate(s) already scored")

    for name in presets:
        if any(b["preset"] == name for b in board):
            print(f"[sweep] {name} already scored; skipping")
            continue
        try:
            r = train_one(rows, name, epochs, out_dir, workers, seed=seed,
                          tag=f"{tag}_{name}", eval_test=False)
        except RuntimeError as e:   # OOM and shape errors both land here
            print(f"[sweep] {name} failed ({type(e).__name__}: {e}); skipping")
            continue
        finally:
            # Free cached GPU memory before the next model.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        p = M.PRESETS[name]
        board.append({
            "preset": name, "family": p.family, "tier": p.tier,
            "params_m": r["params_m"], "img_size": p.img_size,
            "val_f1": r["best"]["val_f1"], "val_acc": r["best"]["val_acc"],
            "secs_per_epoch": round(
                sum(h["secs"] for h in r["history"]) / max(len(r["history"]), 1), 1),
        })
        board_path.write_text(json.dumps({"leaderboard": board,
                                          "complete": False}, indent=2))

    if not board:
        raise RuntimeError("every candidate failed")
    board.sort(key=lambda b: -b["val_f1"])
    winner = board[0]["preset"]

    print(f"\n{'preset':<20}{'family':<8}{'params(M)':>10}{'val_f1':>9}"
          f"{'val_acc':>9}{'s/epoch':>9}")
    for b in board:
        print(f"{b['preset']:<20}{b['family']:<8}{b['params_m']:>10.1f}"
              f"{b['val_f1']:>9.4f}{b['val_acc']:>9.4f}{b['secs_per_epoch']:>9.1f}")
    print(f"\n[sweep] winner on validation: {winner}, evaluating on test")

    final = train_one(rows, winner, epochs, out_dir, workers, seed=seed,
                      tag=f"{tag}_WINNER_{winner}", eval_test=True)
    result = {"leaderboard": board, "winner": winner, "winner_test": final.get("test"),
              "selected_on": "val_f1", "complete": True,
              "note": "test evaluated once, after selection"}
    board_path.write_text(json.dumps(result, indent=2))
    print(f"[ok] {board_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="fine-tune one backbone, or search a family of them")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--preset", default="vit_small",
                    choices=list(M.PRESETS) + list(M.SWEEPS),
                    help="one preset, or a named sweep: " + ", ".join(M.SWEEPS))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="microbatch override for constrained devices")
    ap.add_argument("--accumulate", type=int, default=1,
                    help="microbatches per optimizer step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    rows = sources.read_manifest(a.manifest)
    if a.preset in M.SWEEPS:
        sweep(rows, M.SWEEPS[a.preset], a.epochs, a.out, a.workers, a.seed,
              tag=a.tag or a.preset)
    else:
        train_one(rows, a.preset, a.epochs, a.out, a.workers, seed=a.seed, tag=a.tag,
                  batch_size=a.batch_size, accumulate=a.accumulate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
