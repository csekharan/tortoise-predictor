"""Score the repo's Texas tortoise Mask R-CNN on the 274 hold-out images.

Runs tleimat_original/tortoisemodel.keras over data/texas_tortoise_composite/test/
and scores it the same way zero_shot_gemini.py scores Gemini, so the two can be
compared on the same pictures. An image counts as POSITIVE when the model makes at least one
tortoise detection with confidence above --threshold (default 0.5, the value
tortoise_predictor.py and the paper use). Ground truth comes from manifest.csv:
class "tortoise" is positive, "blank" and "other_animal" are negative.

The composite set has image-level labels only (no outlines), so this is
image-level scoring, not the paper's rule that a predicted box must overlap
the drawn tortoise by IoU >= 0.5.

Every image's highest tortoise score is saved, so the threshold sweep in the
metrics JSON, or any other threshold later, needs no second run. Finished
images are saved as they come in; an interrupted run resumes where it stopped.

Needs torch, torchvision and pillow (CPU builds are enough). The first run
downloads torchvision's COCO Mask R-CNN weights (~170 MB) once, exactly as
utils.get_model_instance_segmentation does, before the tortoise weights are
loaded over them.

Usage:
    python eval_maskrcnn.py                  # whole hold-out set
    python eval_maskrcnn.py --limit 10       # quick smoke test, mixed classes
    python eval_maskrcnn.py --threshold 0.8  # headline metrics at another threshold
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import sys
import time

import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data", "texas_tortoise_composite")
# The original model weights are read in place; none of the original code is imported.
WEIGHTS = os.path.join(REPO, "tleimat_original", "tortoisemodel.keras")
FIELDS = ["file", "truth_class", "truth_species", "source", "truth_positive",
          "max_score", "n_dets_over_0.5", "top_box_xyxy", "width", "height", "seconds"]
SWEEP = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95


def build_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """Same construction as utils.get_model_instance_segmentation(2), copied here
    so this script does not need utils.py's pycocotools and OpenCV imports."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
    # Mask R-CNN computes masks only after boxes and scores are final, so turning
    # the mask branch off changes no score. It saves time and memory: pasting up
    # to 100 full-size masks into a 2048-pixel frame takes over a gigabyte.
    model.roi_heads.mask_roi_pool = None
    model.roi_heads.mask_head = None
    model.roi_heads.mask_predictor = None
    return model.to(device).eval()


def confusion(recs: list[dict], thr: float) -> dict:
    tp = tn = fp = fn = 0
    for r in recs:
        pos = float(r["max_score"]) > thr
        if int(r["truth_positive"]):
            tp, fn = tp + pos, fn + (not pos)
        else:
            fp, tn = fp + pos, tn + (not pos)
    div = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {"threshold": thr, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": div(tp + tn, tp + tn + fp + fn), "precision": div(tp, tp + fp),
            "recall_sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "false_positive_rate": div(fp, fp + tn), "f1": div(2 * tp, 2 * tp + fp + fn)}


def roc_auc(recs: list[dict]) -> float | None:
    """Probability that a random tortoise image outscores a random negative (ties count half)."""
    pos = [float(r["max_score"]) for r in recs if int(r["truth_positive"])]
    neg = [float(r["max_score"]) for r in recs if not int(r["truth_positive"])]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def write_csv(path: str, recs: list[dict]) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(recs)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=DATA, help="texas_tortoise_composite folder")
    ap.add_argument("--weights", default=WEIGHTS, help="tortoise model state dict")
    ap.add_argument("--threshold", type=float, default=0.5, help="confidence cut-off for the headline metrics")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--limit", type=int, default=0, help="only N images, drawn at random across classes (smoke test)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --limit")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results and score every image again")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    with open(os.path.join(args.data, "manifest.csv"), encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.limit]
    order = {r["file"]: i for i, r in enumerate(rows)}

    os.makedirs(args.out, exist_ok=True)
    pred_csv = os.path.join(args.out, "predictions_maskrcnn.csv")
    metrics_json = os.path.join(args.out, "metrics_maskrcnn.json")
    done: dict[str, dict] = {}
    if os.path.exists(pred_csv) and not args.fresh:
        with open(pred_csv, encoding="utf-8", newline="") as f:
            done = {r["file"]: r for r in csv.DictReader(f) if r["file"] in order}
    todo = [r for r in rows if r["file"] not in done]
    print(f"{len(rows)} hold-out images, {len(done)} already scored, {len(todo)} to run on {device}", flush=True)

    model = build_model(args.weights, device) if todo else None
    t_start = time.time()
    try:
        for i, r in enumerate(todo, 1):
            t0 = time.time()
            img = Image.open(os.path.join(args.data, r["file"])).convert("RGB")
            with torch.inference_mode():
                out = model([F.to_tensor(img).to(device)])[0]
            keep = out["labels"] == 1
            scores, boxes = out["scores"][keep].cpu(), out["boxes"][keep].cpu()
            top = int(scores.argmax()) if len(scores) else None
            done[r["file"]] = {
                "file": r["file"], "truth_class": r["class"], "truth_species": r["species"],
                "source": r["source"], "truth_positive": int(r["class"] == "tortoise"),
                "max_score": round(float(scores[top]), 4) if top is not None else 0.0,
                "n_dets_over_0.5": int((scores > 0.5).sum()),
                "top_box_xyxy": " ".join(f"{v:.0f}" for v in boxes[top].tolist()) if top is not None else "",
                "width": img.width, "height": img.height, "seconds": round(time.time() - t0, 2)}
            d = done[r["file"]]
            hit = d["max_score"] > args.threshold
            outcome = ("TP" if hit else "FN") if d["truth_positive"] else ("FP" if hit else "TN")
            print(f"[{len(done)}/{len(rows)}] {outcome:3} {r['class']:12} {r['species']:22} "
                  f"max={d['max_score']:.3f}  {d['seconds']:.1f}s", flush=True)
            if i % 10 == 0:
                write_csv(pred_csv, sorted(done.values(), key=lambda x: order[x["file"]]))
    except KeyboardInterrupt:
        print("\ninterrupted; saving what is done so far", flush=True)
    finally:
        write_csv(pred_csv, sorted(done.values(), key=lambda x: order[x["file"]]))
    elapsed = time.time() - t_start

    recs = [done[r["file"]] for r in rows if r["file"] in done]
    head = confusion(recs, args.threshold)
    secs = sorted(float(r["seconds"]) for r in recs)
    metrics = {
        "model": "tortoisemodel.keras (Mask R-CNN ResNet-50 FPN, Tleimat et al. 2026)",
        "device": str(device), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "images_scored": len(recs), "scoring": "image level: positive if any tortoise detection scores above the threshold",
        **head,
        "roc_auc": roc_auc(recs),
        "median_seconds_per_image": secs[len(secs) // 2] if secs else None,
        "seconds_this_run": round(elapsed, 1),
        "threshold_sweep": [confusion(recs, t) for t in SWEEP],
    }
    # Where the false alarms and misses come from, at the headline threshold.
    by_label: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in recs:
        label = r["truth_class"] if r["truth_class"] != "other_animal" else f"other_animal: {r['truth_species']}"
        src = r["source"].split(" (")[0].split(" et al")[0]
        by_label[f"{label} [{src}]"]["flagged" if float(r["max_score"]) > args.threshold else "not flagged"] += 1
    metrics["flagged_by_true_label_and_source"] = {k: dict(v) for k, v in sorted(by_label.items())}
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n=== tortoisemodel.keras on {len(recs)} hold-out images (threshold {args.threshold}) ===")
    print(f"TP={head['TP']}  TN={head['TN']}  FP={head['FP']}  FN={head['FN']}")
    for k in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1"):
        print(f"{k:19} {head[k]}")
    print(f"roc_auc             {metrics['roc_auc']}")
    print(f"\npredictions: {pred_csv}\nmetrics:     {metrics_json}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
