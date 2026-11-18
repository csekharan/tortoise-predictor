"""Join the Gemini and Mask R-CNN predictions on the 274 hold-out images and break
the results down: what each model flagged, by true label and source; where the
two agree; Mask R-CNN's misses and its highest-scoring false alarms; and its
threshold sweep.

Usage:
    python compare_models.py                       # default result files
    python compare_models.py GEMINI_CSV MASKRCNN_CSV MASKRCNN_JSON
    python compare_models.py results/predictions_gemini-3.8-flash_neutral.csv
"""
import collections
import csv
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
defaults = [os.path.join(RESULTS, n) for n in
            ("predictions_gemini-3.8-flash.csv", "predictions_maskrcnn.csv", "metrics_maskrcnn.json")]
gem_csv, mrc_csv, mrc_json = (sys.argv[1:] + defaults[len(sys.argv) - 1:])[:3]
gem = {r["file"]: r for r in csv.DictReader(open(gem_csv, encoding="utf-8"))}
mrc = {r["file"]: r for r in csv.DictReader(open(mrc_csv, encoding="utf-8"))}
files = [f for f in mrc if f in gem]
print(f"{len(files)} images in both")


def src(r):
    s = r["source"]
    return {"iNaturalist": "iNat", "Wikimedia Commons": "Commons"}.get(s, s[:9])


def label(r):
    return r["truth_class"] if r["truth_class"] != "other_animal" else f"animal:{r['truth_species']}"


print("\n--- flagged as tortoise, by true label and source (Gemini | Mask R-CNN @0.5) ---")
tab = collections.defaultdict(lambda: [0, 0, 0])
for f in files:
    m, g = mrc[f], gem[f]
    k = (label(m), src(m))
    tab[k][0] += 1
    tab[k][1] += int(g["pred_positive"])
    tab[k][2] += float(m["max_score"]) > 0.5
for (lab, s), (n, gp, mp) in sorted(tab.items()):
    print(f"{lab:28} {s:9} n={n:3}  gemini={gp:3}  maskrcnn={mp:3}")

print("\n--- agreement on tortoise call ---")
agree = collections.Counter()
for f in files:
    agree[(int(gem[f]["pred_positive"]), int(float(mrc[f]["max_score"]) > 0.5), int(mrc[f]["truth_positive"]))] += 1
for (g, m, t), n in sorted(agree.items()):
    print(f"gemini={'T' if g else '-'} maskrcnn={'T' if m else '-'} truth={'tortoise' if t else 'negative':8} n={n}")

print("\n--- Mask R-CNN misses (tortoises at <=0.5), lowest scores ---")
miss = sorted((float(mrc[f]["max_score"]), f) for f in files if int(mrc[f]["truth_positive"]) and float(mrc[f]["max_score"]) <= 0.5)
for s, f in miss[:12]:
    print(f"{s:.3f} {f}  gemini={gem[f]['pred_animal']}")
print("\n--- Mask R-CNN top-scoring false alarms ---")
fa = sorted(((float(mrc[f]["max_score"]), f) for f in files if not int(mrc[f]["truth_positive"])), reverse=True)
for s, f in fa[:12]:
    print(f"{s:.3f} {f}  gemini={gem[f]['pred_animal']}")

met = json.load(open(mrc_json, encoding="utf-8"))
print("\n--- Mask R-CNN threshold sweep ---")
for t in met["threshold_sweep"]:
    print(f"thr={t['threshold']:.2f}  TP={t['TP']:3} FP={t['FP']:3} FN={t['FN']:3} TN={t['TN']:3}  "
          f"P={t['precision']}  R={t['recall_sensitivity']}  F1={t['f1']}  FPR={t['false_positive_rate']}")
print("roc_auc", met["roc_auc"], "| median s/img", met["median_seconds_per_image"], "| run s", met["seconds_this_run"])
