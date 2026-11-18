"""Zero-shot animal identification of the 274 hold-out images with Gemini Flash.

For every image in data/texas_tortoise_composite/test/ the model is asked, with no
training and no examples, whether an animal is present and, if so, which one.
A prediction counts as POSITIVE when the model names a tortoise or turtle and
NEGATIVE otherwise (another animal, or no animal). Ground truth comes from
manifest.csv: class "tortoise" is positive, "blank" and "other_animal" are
negative. The script prints TP, TN, FP, FN and the usual rates, and writes
one CSV row per image plus a metrics JSON.

Standard library only, so it runs on any Python 3.10+ without installing
anything. Requests go straight to the Gemini REST API, several at a time, and
finished images are saved as they come in, so an interrupted run resumes
where it stopped and only images that errored are sent again.

Usage (PowerShell: $env:GEMINI_API_KEY="..."; cmd: set GEMINI_API_KEY=...):
    python zero_shot_gemini.py                     # whole hold-out set
    python zero_shot_gemini.py --limit 10          # quick smoke test, mixed classes
    python zero_shot_gemini.py --workers 8         # more parallel requests
    python zero_shot_gemini.py --env-file ..\\some\\.env
    python zero_shot_gemini.py --model gemini-3.8-flash --thinking-level low
    python zero_shot_gemini.py --prompt neutral    # no animal named, no tortoise question
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
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data", "texas_tortoise_composite")
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# --prompt picks one of these. "original" is the prompt of the first run, kept
# word for word so that run can be reproduced. It names "Texas tortoise" first
# among its examples and asks a yes/no tortoise question, both of which can
# nudge the model toward the target class. "neutral" names no animal and asks
# nothing about tortoises: the model only says what it sees, and the tortoise
# decision is made afterwards from the name it gave (POSITIVE_WORDS below).
PROMPTS = {
    "original": (
        "Look at this photograph. Is there an animal in it?\n"
        "If yes, give the animal's common name as precisely as you can "
        "(for example: Texas tortoise, box turtle, nine-banded armadillo, raccoon, "
        "white-tailed deer, cattle, bird, feral hog, coyote).\n"
        "If there is no animal, answer no animal.\n"
        "Return JSON only."
    ),
    "neutral": (
        "Look at this photograph. Is there an animal in it?\n"
        "If yes, give the animal's common name as precisely as you can.\n"
        "If there is no animal, answer none.\n"
        "Return JSON only."
    ),
}

# Structured output: the model must return exactly these fields.
_PRESENT = {"type": "BOOLEAN"}
_ANIMAL = {"type": "STRING", "description": "Common name of the animal, or \"none\" if no animal."}
SCHEMAS = {
    "original": {
        "type": "OBJECT",
        "properties": {"animal_present": _PRESENT, "animal": _ANIMAL,
                       "is_tortoise_or_turtle": {"type": "BOOLEAN"}},
        "required": ["animal_present", "animal", "is_tortoise_or_turtle"],
    },
    "neutral": {
        "type": "OBJECT",
        "properties": {"animal_present": _PRESENT, "animal": _ANIMAL},
        "required": ["animal_present", "animal"],
    },
}

# Names that count as the positive class when the model's own flag is missing.
POSITIVE_WORDS = re.compile(
    r"tortoise|turtle|terrapin|gopherus|testudin|chelon|cooter|slider",
    re.IGNORECASE)

MAGIC = {b"\xff\xd8\xff": "image/jpeg", b"\x89PNG": "image/png",
         b"RIFF": "image/webp", b"GIF8": "image/gif"}

# HTTP codes that mean the request itself is wrong (bad key, bad model id,
# malformed body): retrying or sending the other 273 images would not help.
FATAL_HTTP = {400, 401, 403, 404}
RETRY_HTTP = {429, 500, 502, 503, 504}
# 402 = prepaid credits depleted. Seen to come and go for minutes after a
# top-up, so it is retried with long waits; if it persists, the run stops.
BILLING_HTTP = 402
BILLING_WAITS = (15, 30, 60, 90, 120)

FIELDS = ["file", "truth_class", "truth_species", "truth_positive",
          "pred_animal_present", "pred_animal", "pred_is_tortoise_or_turtle",
          "pred_positive", "outcome", "seconds", "error", "raw"]


class FatalAPIError(RuntimeError):
    """An error that will recur for every image; stop the run."""


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
                if line.startswith("export "):
                    line = line[7:]
                if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        sys.exit(f"No GEMINI_API_KEY or GOOGLE_API_KEY line in {args.env_file}")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY), or pass --env-file")
    return key


class Gemini:
    """Thin REST client: one image in, one JSON object out, with retries."""

    def __init__(self, key: str, model: str, temperature: float, thinking_level: str | None,
                 prompt: str = "original", timeout: float = 120.0, tries: int = 6):
        self.url = API.format(model=model)
        self.key = key
        self.prompt = PROMPTS[prompt]
        self.timeout = timeout
        self.tries = tries
        self.usage = collections.Counter()
        self.lock = threading.Lock()
        self.gen_config: dict = {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMAS[prompt],
            # Generous: on thinking models the thoughts count against this
            # limit, and a tight cap returns an empty answer (MAX_TOKENS).
            "maxOutputTokens": 4096,
        }
        if thinking_level:
            self.gen_config["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}

    def classify(self, path: str) -> dict:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        body = json.dumps({
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": mime_of(path), "data": data}},
                {"text": self.prompt},
            ]}],
            "generationConfig": self.gen_config,
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
                if e.code in FATAL_HTTP:
                    raise FatalAPIError(f"HTTP {e.code}: {detail}") from None
                if e.code == BILLING_HTTP:
                    if attempt <= len(BILLING_WAITS):
                        time.sleep(BILLING_WAITS[attempt - 1] + random.uniform(0, 3))
                        continue
                    raise FatalAPIError(f"HTTP {e.code} after {attempt} tries over "
                                        f"{sum(BILLING_WAITS) // 60} minutes: {detail}") from None
                if e.code in RETRY_HTTP and attempt < self.tries:
                    wait = float(e.headers.get("Retry-After") or 0) or delay
                    time.sleep(min(wait, 120) + random.uniform(0, 1))
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
        parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text:
            raise RuntimeError(f"empty reply: finishReason={cand.get('finishReason')} "
                               f"promptFeedback={reply.get('promptFeedback')}")
        try:
            out = json.loads(text)
            if not isinstance(out, dict):
                raise ValueError
        except ValueError:
            # Schema-constrained output should always be JSON; keep the raw text if not.
            out = {"animal_present": None, "animal": text.strip(), "is_tortoise_or_turtle": None}
        out["_raw"] = text.strip()
        return out


def predicted_positive(out: dict) -> bool:
    """Tortoise or turtle, by the model's flag or by the name it gave.
    'No animal present' wins over a stray tortoise flag."""
    if out.get("animal_present") is False:
        return False
    if out.get("is_tortoise_or_turtle") is True:
        return True
    return bool(POSITIVE_WORDS.search(out.get("animal") or ""))


def write_csv(path: str, recs: list[dict]) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(recs)
    os.replace(tmp, path)


def main() -> None:
    # Model answers can contain non-ASCII letters; never let the console encoding kill a worker.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=DATA, help="texas_tortoise_composite folder")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    ap.add_argument("--limit", type=int, default=0, help="only N images, drawn at random across classes (smoke test)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --limit")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--env-file", help="read the API key from a KEY=VALUE file")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"],
                    help="send thinkingConfig.thinkingLevel (Gemini 3 models); default: not sent, model default")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="original",
                    help="original: the first run's prompt (lists Texas tortoise first, asks a tortoise yes/no); "
                         "neutral: names no animal, tortoise decided from the name the model gives")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results and score every image again")
    args = ap.parse_args()

    key = load_key(args)
    manifest = os.path.join(args.data, "manifest.csv")
    with open(manifest, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if not rows:
        sys.exit(f"no test rows in {manifest}")
    missing = [r["file"] for r in rows if not os.path.exists(os.path.join(args.data, r["file"]))]
    if missing:
        sys.exit(f"{len(missing)} test images listed in the manifest are not on disk, e.g. {missing[0]}")
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.limit]
    order = {r["file"]: i for i, r in enumerate(rows)}

    os.makedirs(args.out, exist_ok=True)
    # The original prompt keeps the first run's file names; other prompts get a suffix.
    tag = re.sub(r"[^A-Za-z0-9.-]+", "_", args.model) + ("" if args.prompt == "original" else f"_{args.prompt}")
    pred_csv = os.path.join(args.out, f"predictions_{tag}.csv")
    metrics_json = os.path.join(args.out, f"metrics_{tag}.json")

    # Resume: keep every earlier answer that did not error.
    done: dict[str, dict] = {}
    if os.path.exists(pred_csv) and not args.fresh:
        with open(pred_csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["outcome"] != "ERROR" and r["file"] in order:
                    done[r["file"]] = r
    todo = [r for r in rows if r["file"] not in done]
    print(f"{len(rows)} hold-out images, {len(done)} already answered, {len(todo)} to send "
          f"to {args.model} ({args.prompt} prompt) with {args.workers} workers", flush=True)

    client = Gemini(key, args.model, args.temperature, args.thinking_level, args.prompt)
    lock = threading.Lock()
    stop = threading.Event()
    fatal: list[str] = []
    counter = [len(done)]

    def save_all() -> None:
        recs = sorted(done.values(), key=lambda r: order[r["file"]])
        write_csv(pred_csv, recs)

    def work(r: dict) -> None:
        if stop.is_set():
            return
        truth_pos = r["class"] == "tortoise"
        rec = {"file": r["file"], "truth_class": r["class"], "truth_species": r["species"],
               "truth_positive": int(truth_pos), "error": "", "raw": ""}
        t0 = time.time()
        try:
            out = client.classify(os.path.join(args.data, r["file"]))
            pos = predicted_positive(out)
            rec.update(pred_animal_present=out.get("animal_present"),
                       pred_animal=out.get("animal"),
                       pred_is_tortoise_or_turtle=out.get("is_tortoise_or_turtle"),
                       pred_positive=int(pos), raw=out["_raw"],
                       outcome=("TP" if pos else "FN") if truth_pos else ("FP" if pos else "TN"))
        except FatalAPIError as e:
            stop.set()
            with lock:
                fatal.append(str(e))
            return
        except Exception as e:  # noqa: BLE001 - record it; the next run retries it
            rec.update(pred_animal_present="", pred_animal="", pred_is_tortoise_or_turtle="",
                       pred_positive="", outcome="ERROR", error=f"{type(e).__name__}: {e}"[:500])
        rec["seconds"] = round(time.time() - t0, 2)
        with lock:
            done[r["file"]] = rec
            counter[0] += 1
            if counter[0] % 10 == 0:
                save_all()
            shown = rec["pred_animal"] if rec["outcome"] != "ERROR" else rec["error"][:70]
            print(f"[{counter[0]}/{len(rows)}] {rec['outcome']:5} {r['class']:12} {r['species']:22} -> {shown}",
                  flush=True)

    t_start = time.time()
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
    except KeyboardInterrupt:
        stop.set()
        print("\ninterrupted; saving what is done so far", flush=True)
    finally:
        save_all()
    elapsed = time.time() - t_start
    if fatal:
        sys.exit(f"\nStopped: the API rejected the request, so every image would fail.\n{fatal[0]}\n"
                 f"Check the key, the model id ({args.model}) and the account's billing. "
                 f"Answers so far are in {pred_csv}.")

    # Metrics over every image that has an answer.
    recs = [done[r["file"]] for r in rows if r["file"] in done and done[r["file"]]["outcome"] != "ERROR"]
    errors = len(rows) - len(recs)
    c = collections.Counter(r["outcome"] for r in recs)
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    div = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    metrics = {
        "model": args.model, "prompt_variant": args.prompt, "prompt": PROMPTS[args.prompt],
        "temperature": args.temperature, "thinking_level": args.thinking_level,
        "images_scored": len(recs), "errors": errors,
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
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n=== Zero-shot results: {args.model}, {args.prompt} prompt ===")
    print(f"scored {len(recs)} of {len(rows)} images" + (f", {errors} errors (run again to retry them)" if errors else ""))
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
