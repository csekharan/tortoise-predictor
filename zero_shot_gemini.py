"""Zero-shot animal identification of the 274 hold-out images with Gemini Flash.

For every image in texas_tortoise_composite/test/ the model is asked, with no
training and no examples, whether an animal is present and, if so, which one.
A prediction counts as POSITIVE when the model names a tortoise or turtle and
NEGATIVE otherwise (another animal, or no animal). Ground truth comes from
manifest.csv: class "tortoise" is positive, "blank" and "other_animal" are
negative. The script prints TP, TN, FP, FN and the usual rates, and writes
one CSV row per image plus a metrics JSON.

Standard library only, so it runs on any Python 3.10+ without installing
anything. Requests go straight to the Gemini REST API, several at a time, and
finished images are appended to the CSV as they come in, so an interrupted run
resumes where it stopped.

Usage:
    set GEMINI_API_KEY=...          (or GOOGLE_API_KEY, or --env-file path/.env)
    python zero_shot_gemini.py                     # whole hold-out set
    python zero_shot_gemini.py --limit 10          # quick smoke test
    python zero_shot_gemini.py --workers 8         # more parallel requests
    python zero_shot_gemini.py --model gemini-3.8-flash --out zero_shot_results
"""
from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures as cf
import csv
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "texas_tortoise_composite")
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = (
    "Look at this photograph. Is there an animal in it?\n"
    "If yes, give the animal's common name as precisely as you can "
    "(for example: Texas tortoise, box turtle, nine-banded armadillo, raccoon, "
    "white-tailed deer, cattle, bird, feral hog, coyote).\n"
    "If there is no animal, answer no animal.\n"
    "Return JSON only."
)

# Structured output: the model must return exactly these fields.
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "animal_present": {"type": "BOOLEAN"},
        "animal": {"type": "STRING",
                   "description": "Common name of the animal, or \"none\" if no animal."},
        "is_tortoise_or_turtle": {"type": "BOOLEAN"},
    },
    "required": ["animal_present", "animal", "is_tortoise_or_turtle"],
}

POSITIVE_WORDS = re.compile(
    r"tortoise|turtle|terrapin|gopherus|testudin|chelon|cooter|slider|snapper",
    re.IGNORECASE)

MAGIC = {b"\xff\xd8\xff": "image/jpeg", b"\x89PNG": "image/png",
         b"RIFF": "image/webp", b"GIF8": "image/gif"}


def mime_of(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(4)
    for magic, mime in MAGIC.items():
        if head.startswith(magic):
            return mime
    return "image/jpeg"


def load_key(args) -> str:
    if args.env_file:
        with open(args.env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        sys.exit(f"No GEMINI_API_KEY or GOOGLE_API_KEY line in {args.env_file}")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY), or pass --env-file")
    return key


class Gemini:
    """Thin REST client: one image in, one JSON object out, with retries."""

    def __init__(self, key: str, model: str, timeout: float = 120.0, tries: int = 6):
        self.url = API.format(model=model)
        self.key = key
        self.timeout = timeout
        self.tries = tries
        self.usage = collections.Counter()
        self.lock = threading.Lock()

    def classify(self, path: str) -> dict:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        body = json.dumps({
            "contents": [{"role": "user", "parts": [
                {"inline_data": {"mime_type": mime_of(path), "data": data}},
                {"text": PROMPT},
            ]}],
            "generationConfig": {
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": SCHEMA,
                "max_output_tokens": 200,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.key})

        delay = 2.0
        for attempt in range(1, self.tries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    reply = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.tries:
                    wait = float(e.headers.get("Retry-After") or 0) or delay
                    time.sleep(wait + random.uniform(0, 1))
                    delay = min(delay * 2, 60)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt < self.tries:
                    time.sleep(delay + random.uniform(0, 1))
                    delay = min(delay * 2, 60)
                    continue
                raise

        with self.lock:
            for k, v in (reply.get("usageMetadata") or {}).items():
                if isinstance(v, int):
                    self.usage[k] += v
        cand = (reply.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
        if not text:
            raise RuntimeError(f"empty reply: finishReason={cand.get('finishReason')} "
                               f"promptFeedback={reply.get('promptFeedback')}")
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            # Schema-constrained output should always be JSON; keep the raw text if not.
            out = {"animal_present": None, "animal": text.strip(), "is_tortoise_or_turtle": None}
        out["_raw"] = text.strip()
        return out


def predicted_positive(out: dict) -> bool:
    """Tortoise or turtle, by either the model's flag or the name it gave."""
    if out.get("is_tortoise_or_turtle") is True:
        return True
    name = out.get("animal") or ""
    return bool(out.get("animal_present")) and bool(POSITIVE_WORDS.search(name))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=DATA, help="texas_tortoise_composite folder")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    ap.add_argument("--out", default=os.path.join(HERE, "zero_shot_results"))
    ap.add_argument("--env-file", help="read the API key from a KEY=VALUE file")
    ap.add_argument("--seed", type=int, default=0, help="shuffle order with --limit so a smoke test mixes classes")
    args = ap.parse_args()

    key = load_key(args)
    manifest = os.path.join(args.data, "manifest.csv")
    with open(manifest, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if not rows:
        sys.exit(f"no test rows in {manifest}")
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9.-]+", "_", args.model)
    pred_csv = os.path.join(args.out, f"predictions_{tag}.csv")
    metrics_json = os.path.join(args.out, f"metrics_{tag}.json")
    fields = ["file", "truth_class", "truth_species", "truth_positive",
              "pred_animal_present", "pred_animal", "pred_is_tortoise_or_turtle",
              "pred_positive", "outcome", "seconds", "error", "raw"]

    # Resume: skip images already answered without error.
    done: dict[str, dict] = {}
    if os.path.exists(pred_csv):
        with open(pred_csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if not r["error"]:
                    done[r["file"]] = r
    todo = [r for r in rows if r["file"] not in done]
    print(f"{len(rows)} hold-out images, {len(done)} already done, {len(todo)} to send "
          f"to {args.model} with {args.workers} workers", flush=True)

    client = Gemini(key, args.model)
    lock = threading.Lock()
    new_file = not os.path.exists(pred_csv)
    fout = open(pred_csv, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=fields)
    if new_file:
        writer.writeheader()
        fout.flush()

    def work(r: dict) -> dict:
        path = os.path.join(args.data, r["file"])
        truth_pos = r["class"] == "tortoise"
        t0 = time.time()
        rec = {"file": r["file"], "truth_class": r["class"], "truth_species": r["species"],
               "truth_positive": int(truth_pos), "error": "", "raw": ""}
        try:
            out = client.classify(path)
            pos = predicted_positive(out)
            rec.update(pred_animal_present=out.get("animal_present"),
                       pred_animal=out.get("animal"),
                       pred_is_tortoise_or_turtle=out.get("is_tortoise_or_turtle"),
                       pred_positive=int(pos), raw=out["_raw"],
                       outcome=("TP" if pos else "FN") if truth_pos else ("FP" if pos else "TN"))
        except Exception as e:  # noqa: BLE001 - record and move on
            rec.update(pred_animal_present="", pred_animal="", pred_is_tortoise_or_turtle="",
                       pred_positive="", outcome="ERROR", error=f"{type(e).__name__}: {e}"[:500])
        rec["seconds"] = round(time.time() - t0, 2)
        with lock:
            writer.writerow(rec)
            fout.flush()
            done[r["file"]] = rec
            n = len(done)
            print(f"[{n}/{len(rows)}] {rec['outcome']:5} {r['class']:12} {r['species']:22} "
                  f"-> {rec['pred_animal'] or rec['error'][:60]}", flush=True)
        return rec

    t_start = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fout.close()
    elapsed = time.time() - t_start

    # Metrics over every image that has an answer.
    recs = [done[r["file"]] for r in rows
            if r["file"] in done and done[r["file"]]["outcome"] != "ERROR"]
    errors = len(rows) - len(recs)
    c = collections.Counter(r["outcome"] for r in recs)
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    div = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    metrics = {
        "model": args.model, "images_scored": len(recs), "errors": errors,
        "positive": "tortoise or turtle", "negative": "any other animal, or no animal",
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "precision": div(tp, tp + fp),
        "recall_sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "seconds_this_run": round(elapsed, 1),
        "token_usage_this_run": dict(client.usage),
    }
    # What the model called each true class / species.
    by_truth: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in recs:
        label = r["truth_class"] if r["truth_class"] != "other_animal" else f"other_animal: {r['truth_species']}"
        by_truth[label][(r["pred_animal"] or "").strip().lower() or "(none)"] += 1
    metrics["model_answers_by_true_label"] = {k: dict(v.most_common()) for k, v in sorted(by_truth.items())}
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Zero-shot results:", args.model, "===")
    print(f"scored {len(recs)} of {len(rows)} images" + (f", {errors} errors (rerun to retry)" if errors else ""))
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    for k in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1"):
        print(f"{k:19} {metrics[k]}")
    if fn:
        print("\nTortoises the model missed were called:",
              dict(collections.Counter((r["pred_animal"] or "").lower() for r in recs if r["outcome"] == "FN").most_common()))
    if fp:
        print("Non-tortoises called tortoise/turtle:",
              dict(collections.Counter(f"{r['truth_class']}/{r['truth_species']}" for r in recs if r["outcome"] == "FP").most_common()))
    print(f"\npredictions: {pred_csv}\nmetrics:     {metrics_json}")


if __name__ == "__main__":
    main()
