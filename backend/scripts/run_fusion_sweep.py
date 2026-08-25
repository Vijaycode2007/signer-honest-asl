"""Cache matched A/B/C predictions, select one fusion ratio, and score `mine`.

Run after the three geometry bundles and A/B pixel checkpoints exist. Protocol C
reuses the deployed DeiT III checkpoint. Ratio selection sees A/B validation and
a group-disjoint 30% tuning carve from the cross-corpus holdout; reported C
final metrics use the untouched remaining 70%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from slr import data, evaluate, fusion, landmarks, model as M, sources  # noqa: E402

OUT = ROOT / "runs" / "fusion"
MANIFESTS = {p: ROOT / "dataset" / "manifests" / f"fusion_{p}.csv" for p in "ABC"}
PIXEL = {
    "A": OUT / "A_pixel" / "best.pt",
    "B": OUT / "B_pixel" / "best.pt",
    "C": ROOT / "runs" / "deploy_deit3_base" / "best.pt",
}
GEOMETRY = {p: OUT / f"{p}_geometry.pt" for p in "ABC"}
CACHE = ROOT / "dataset" / "landmarks" / "cross.npz"
MINE_MANIFEST = ROOT / "dataset" / "manifests" / "mine.csv"
MINE_CACHE = ROOT / "dataset" / "landmarks" / "mine.npz"


@torch.no_grad()
def collect_pixel(net: torch.nn.Module, rows: list[dict], img_size: int,
                  dev: torch.device, batch: int = 16) -> np.ndarray:
    loader = DataLoader(data.SignDataset(rows, img_size, train=False),
                        # Managed macOS sandboxes can forbid torch_shm_manager.
                        # Evaluation is sequential and GPU-bound, so an in-process
                        # loader is both portable and sufficient here.
                        batch_size=batch, shuffle=False, num_workers=0)
    logits = []
    net.eval()
    for x, _ in loader:
        logits.append(net(x.to(dev)).float().cpu())
    return torch.cat(logits).numpy()


def cache_protocol(protocol: str, landmark_cache: dict) -> dict:
    rows = sources.read_manifest(MANIFESTS[protocol])
    blob = torch.load(PIXEL[protocol], map_location="cpu", weights_only=False)
    if list(blob["classes"]) != list(sources.CLASSES):
        raise ValueError(f"{protocol} pixel class order differs from the project vocabulary")
    dev = M.device()
    net = M.build(blob["preset"], len(blob["classes"]), pretrained=False).to(dev)
    net.load_state_dict(blob["state_dict"])
    split_rows = {name: [r for r in rows if r["split"] == name]
                  for name in ("val", "test")}
    raw_logits = {name: collect_pixel(net, split_rows[name], blob["img_size"], dev)
                  for name in ("val", "test")}
    val_truth = torch.tensor([sources.CLASS_TO_IDX[r["label"]]
                              for r in split_rows["val"]])
    temperature = evaluate.fit_temperature(
        torch.from_numpy(raw_logits["val"]), val_truth)

    geometry = fusion.load_geometry(GEOMETRY[protocol])
    if geometry.classes != list(sources.CLASSES):
        raise ValueError(f"{protocol} geometry class order differs from pixel classes")

    saved: dict[str, np.ndarray] = {"pixel_temperature": np.array(temperature)}
    for name in ("val", "test"):
        subset = split_rows[name]
        cached = fusion.cached_landmarks_for_rows(landmark_cache, subset)
        saved[f"{name}_pixel"] = torch.softmax(
            torch.from_numpy(raw_logits[name]) / temperature, dim=1).numpy()
        saved[f"{name}_geometry"] = fusion.geometry_probabilities(
            geometry, cached["lm"], cached["found"], cached["side"])
        saved[f"{name}_truth"] = np.array(
            [sources.CLASS_TO_IDX[r["label"]] for r in subset], dtype=np.int64)
        saved[f"{name}_path"] = np.array([r["path"] for r in subset])

    if protocol == "C":
        mine = sources.read_manifest(MINE_MANIFEST)
        saved["mine_pixel"] = torch.softmax(
            torch.from_numpy(collect_pixel(net, mine, blob["img_size"], dev)) / temperature,
            dim=1).numpy()
        saved["mine_path"] = np.array([r["path"] for r in mine])
        saved["mine_truth"] = np.array(
            [sources.CLASS_TO_IDX[r["label"]] for r in mine], dtype=np.int64)

    path = OUT / f"{protocol}_predictions.npz"
    np.savez_compressed(path, **saved)
    print(f"[fusion] cached {protocol} predictions -> {path} (T={temperature:.4f})")
    del net
    if dev.type == "mps":
        torch.mps.empty_cache()
    return saved


def main() -> int:
    missing = [str(p) for p in [*MANIFESTS.values(), *PIXEL.values(),
                                *GEOMETRY.values(), CACHE, MINE_MANIFEST, MINE_CACHE]
               if not p.exists()]
    if missing:
        raise SystemExit("missing fusion inputs:\n  " + "\n  ".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    landmark_cache = landmarks.load_cache(CACHE)
    predictions = {p: cache_protocol(p, landmark_cache) for p in "ABC"}

    c_rows = sources.read_manifest(MANIFESTS["C"])
    c_test_rows = [r for r in c_rows if r["split"] == "test"]
    tune_rows, final_rows = fusion.partition_tuning_rows(
        c_test_rows, tune_fraction=0.30, seed=0)
    tune_paths = {r["path"] for r in tune_rows}
    final_paths = {r["path"] for r in final_rows}
    c_paths = predictions["C"]["test_path"]
    tune_mask = np.array([p in tune_paths for p in c_paths])
    final_mask = np.array([p in final_paths for p in c_paths])
    if (tune_mask & final_mask).any() or not (tune_mask | final_mask).all():
        raise AssertionError("cross-corpus tuning/final partition is not exhaustive and disjoint")

    selection = {
        "A": {"pixel": predictions["A"]["val_pixel"],
              "geometry": predictions["A"]["val_geometry"],
              "truth": predictions["A"]["val_truth"]},
        "B": {"pixel": predictions["B"]["val_pixel"],
              "geometry": predictions["B"]["val_geometry"],
              "truth": predictions["B"]["val_truth"]},
        "C": {"pixel": predictions["C"]["test_pixel"][tune_mask],
              "geometry": predictions["C"]["test_geometry"][tune_mask],
              "truth": predictions["C"]["test_truth"][tune_mask]},
    }
    sweep = fusion.sweep_probabilities(selection)
    winner = fusion.choose_ratio(sweep, max_ab_regression=0.01)
    weight = winner["pixel_weight"]
    c_tune_fused = fusion.fuse_probabilities(
        selection["C"]["pixel"], selection["C"]["geometry"], weight)
    operating_point = evaluate.risk_coverage(
        torch.from_numpy(c_tune_fused), torch.from_numpy(selection["C"]["truth"]),
        coverages=(0.9,),
    )[0]

    final_inputs = {
        "A": {"pixel": predictions["A"]["test_pixel"],
              "geometry": predictions["A"]["test_geometry"],
              "truth": predictions["A"]["test_truth"]},
        "B": {"pixel": predictions["B"]["test_pixel"],
              "geometry": predictions["B"]["test_geometry"],
              "truth": predictions["B"]["test_truth"]},
        "C": {"pixel": predictions["C"]["test_pixel"][final_mask],
              "geometry": predictions["C"]["test_geometry"][final_mask],
              "truth": predictions["C"]["test_truth"][final_mask]},
    }
    final = {}
    for p, item in final_inputs.items():
        final[p] = {
            "pixel": fusion.classification_metrics(item["pixel"], item["truth"]),
            "geometry": fusion.classification_metrics(item["geometry"], item["truth"]),
            "fused": fusion.classification_metrics(
                fusion.fuse_probabilities(item["pixel"], item["geometry"], weight),
                item["truth"]),
        }

    mine_rows = sources.read_manifest(MINE_MANIFEST)
    mine_cache = landmarks.load_cache(MINE_CACHE)
    mine_geo_cache = fusion.cached_landmarks_for_rows(mine_cache, mine_rows)
    mine_geometry = fusion.geometry_probabilities(
        fusion.load_geometry(GEOMETRY["C"]), mine_geo_cache["lm"],
        mine_geo_cache["found"], mine_geo_cache["side"])
    mine_pixel = predictions["C"]["mine_pixel"]
    mine_fused = fusion.fuse_probabilities(mine_pixel, mine_geometry, weight)
    mine = []
    for i, row in enumerate(mine_rows):
        top = np.argsort(mine_fused[i])[::-1][:3]
        mine.append({
            "path": row["path"], "truth": row["label"],
            "pixel": sources.CLASSES[int(mine_pixel[i].argmax())],
            "geometry": sources.CLASSES[int(mine_geometry[i].argmax())],
            "fused": sources.CLASSES[int(mine_fused[i].argmax())],
            "correct": sources.CLASSES[int(mine_fused[i].argmax())] == row["label"],
            "top3": [{"label": sources.CLASSES[int(j)], "p": float(mine_fused[i, j])}
                     for j in top],
        })

    report = {
        "selection_rule": "maximize C tuning macro-F1; A/B may lose at most 0.01 vs pixel",
        "pixel_weight": weight,
        "geometry_weight": 1.0 - weight,
        "selection": winner,
        "operating_point": {**operating_point, "picked_on": "C_group_disjoint_tuning"},
        "cross_corpus_partition": {
            "tuning_n": int(tune_mask.sum()), "final_n": int(final_mask.sum()),
            "shared_groups": len({r["group"] for r in tune_rows} &
                                 {r["group"] for r in final_rows}),
        },
        "final": final,
        "cross_corpus_full_descriptive": {
            "pixel": fusion.classification_metrics(
                predictions["C"]["test_pixel"], predictions["C"]["test_truth"]),
            "geometry": fusion.classification_metrics(
                predictions["C"]["test_geometry"], predictions["C"]["test_truth"]),
            "fused": fusion.classification_metrics(
                fusion.fuse_probabilities(predictions["C"]["test_pixel"],
                                          predictions["C"]["test_geometry"], weight),
                predictions["C"]["test_truth"]),
        },
        "mine": {
            "metrics": {
                "pixel": fusion.classification_metrics(
                    mine_pixel, predictions["C"]["mine_truth"]),
                "geometry": fusion.classification_metrics(
                    mine_geometry, predictions["C"]["mine_truth"]),
                "fused": fusion.classification_metrics(
                    mine_fused, predictions["C"]["mine_truth"]),
            },
            "photos": mine,
        },
        "all_ratios": sweep,
    }
    path = OUT / "fusion_report.json"
    path.write_text(json.dumps(report, indent=2))

    # One self-contained serving artifact. The standalone pixel bundle remains
    # untouched so the selected fusion can always be compared or rolled back.
    pixel_bundle_path = ROOT / "app" / "weights" / "deit3_base.pt"
    fused_bundle_path = ROOT / "app" / "weights" / "deit3_base_fused.pt"
    pixel_bundle = torch.load(pixel_bundle_path, map_location="cpu", weights_only=False)
    geometry_blob = torch.load(GEOMETRY["C"], map_location="cpu", weights_only=False)
    if list(pixel_bundle["classes"]) != list(geometry_blob["classes"]):
        raise ValueError("deploy pixel and C geometry bundles use different class orders")
    pixel_bundle["fusion"] = {
        "pixel_weight": weight,
        "geometry": {k: geometry_blob[k] for k in (
            "state_dict", "classes", "mean", "std", "hidden", "feature", "temperature")},
    }
    pixel_bundle["threshold"] = operating_point["threshold"]
    pixel_bundle["coverage"] = operating_point["coverage"]
    pixel_bundle["fusion_metrics"] = {
        "selection": winner,
        "final": final,
        "operating_point": report["operating_point"],
    }
    torch.save(pixel_bundle, fused_bundle_path)
    print(f"[fusion] winner pixel:geometry = {weight:.2f}:{1-weight:.2f}")
    print(json.dumps({"final": final, "mine": report["mine"]["metrics"]}, indent=2))
    print(f"[fusion] report -> {path}")
    print(f"[fusion] serving bundle -> {fused_bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
