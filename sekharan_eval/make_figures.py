"""Draw the README figures from the prediction files in results/.

Writes four charts, each as a light and a dark SVG, to figures/:
    confusion-{light,dark}.svg     confusion matrices: Gemini (two prompts) and Mask R-CNN (two thresholds)
    false-alarms-{light,dark}.svg  which non-tortoise images each method called tortoise
    tradeoff-{light,dark}.svg      Mask R-CNN's threshold curve against Gemini's two points
    species-{light,dark}.svg       what Gemini called the tortoises under each prompt
and prints the numbers the README quotes (counts, rates with 95% Clopper-Pearson
intervals, exact McNemar tests on the non-tortoise images, species names).

Standard library only. Usage:
    python make_figures.py
"""
from __future__ import annotations

import collections
import csv
import math
import os
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
GEMINI_FILES = {"gem": "predictions_gemini-3.8-flash.csv", "neu": "predictions_gemini-3.8-flash_neutral.csv"}

# The manifest labels this photo a tortoise, but it shows a bulldozer and a
# pickup truck and no tortoise, so it is scored as a negative here.
LABEL_FIXES = {"test/tortoise/inat-tortoise_inat50015197_p79414426.jpg": 0}

# Colour follows the method in every chart: blue = Gemini with the original
# prompt, aqua = Gemini with the neutral prompt, orange = Mask R-CNN. These are
# slots 1-3 of the reference palette, validated all-pairs in both modes.
THEMES = {
    "light": dict(surface="#fcfcfb", ring="#0b0b0b", ring_op=0.10, ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  gem="#2a78d6", neu="#1baf7a", mrc="#eb6834"),
    "dark": dict(surface="#1a1a19", ring="#ffffff", ring_op=0.10, ink="#f0efec", ink2="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835",
                 gem="#3987e5", neu="#199e70", mrc="#d95926"),
}
NAMES = {"gem": "Gemini, original prompt", "neu": "Gemini, neutral prompt", "mrc": "Mask R-CNN, threshold 0.5"}
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
W = 900
PAD = 32


# ---------------------------------------------------------------- data

def load() -> list[dict]:
    gem = {k: {r["file"]: r for r in csv.DictReader(open(os.path.join(RESULTS, f), encoding="utf-8"))}
           for k, f in GEMINI_FILES.items()}
    rows = []
    for r in csv.DictReader(open(os.path.join(RESULTS, "predictions_maskrcnn.csv"), encoding="utf-8")):
        f = r["file"]
        src = "South Texas" if "Kutugata" in r["source"] else ("Florida" if "NACTI" in r["source"] else "iNaturalist")
        row = {"file": f, "truth": LABEL_FIXES.get(f, int(r["truth_positive"])),
               "cls": r["truth_class"], "species": r["truth_species"], "src": src, "score": float(r["max_score"])}
        for k, g in gem.items():
            row[k] = int(g[f]["pred_positive"])
            row[k + "_name"] = g[f]["pred_animal"]
        rows.append(row)
    return rows


def confusion(rows, called) -> dict:
    c = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    for r in rows:
        p = called(r)
        c[("TP" if p else "FN") if r["truth"] else ("FP" if p else "TN")] += 1
    return c


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    def cdf(x, p):  # P(X <= x), X ~ Binomial(n, p)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))

    def solve(f, target):  # f increasing in p
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - cdf(k - 1, p), alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: 1 - cdf(k, p), 1 - alpha / 2)
    return lower, upper


def mcnemar_exact(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def species_group(name: str) -> str:
    n = name.lower()
    if "texas tortoise" in n:
        return "Texas tortoise"
    if n.startswith("tortoise"):
        return "tortoise"
    for k in ("desert tortoise", "gopher tortoise", "red-footed tortoise"):
        if k in n:
            return k
    if "turtle" in n or "terrapin" in n:
        return "turtle"
    return "other tortoise" if "tortoise" in n else "not a tortoise"


# ---------------------------------------------------------------- svg helpers

def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, t):
    ra, rb = hex_rgb(a), hex_rgb(b)
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ra, rb))


def text(x, y, s, size=13, fill="#000", weight=400, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" {extra}>{escape(s)}</text>')


def hbar(x0, y, w, h, fill):
    """Horizontal bar: square at the baseline (left), 4px rounded data-end (right)."""
    if w <= 0:
        return ""
    r = min(4, w, h / 2)
    return (f'<path d="M{x0:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} '
            f'a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.1f} z" fill="{fill}"/>')


def svg(theme, height, body, title, desc):
    t = THEMES[theme]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" '
            f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
            f'<title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>\n'
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="12" fill="{t["surface"]}" '
            f'stroke="{t["ring"]}" stroke-opacity="{t["ring_op"]}"/>\n' + "\n".join(body) + "\n</svg>\n")


def heading(t, title, subtitle):
    return [text(PAD, 44, title, 19, t["ink"], 600), text(PAD, 68, subtitle, 13.5, t["ink2"])]


def legend(t, items, y, x=PAD, step=None):
    """items: (colour key, label, 'dot' | 'square' | 'line'). Returns svg parts."""
    out = []
    for i, (key, label, shape) in enumerate(items):
        lx = x + (step * i if step else 0)
        if shape == "line":
            out.append(f'<line x1="{lx}" y1="{y - 4}" x2="{lx + 16}" y2="{y - 4}" stroke="{t[key]}" '
                       f'stroke-width="2" stroke-linecap="round"/>')
            tx = lx + 24
        elif shape == "dot":
            out.append(f'<circle cx="{lx + 6}" cy="{y - 4}" r="5" fill="{t[key]}"/>')
            tx = lx + 18
        else:
            out.append(f'<rect x="{lx}" y="{y - 10}" width="12" height="12" rx="3" fill="{t[key]}"/>')
            tx = lx + 20
        out.append(text(tx, y, label, 12.5, t["ink2"]))
    return out


# ---------------------------------------------------------------- figure 1

def fig_confusion(theme, grid, n_pos, n_neg):
    """grid: rows of panels; each panel is (name, subtitle, confusion, colour key)."""
    t = THEMES[theme]
    body = heading(t, f"Both methods find all {n_pos} tortoises; the false alarms set them apart",
                   f"Hold-out set: {n_pos} tortoise photos and {n_neg} images without one. "
                   "Shade shows each cell's share of its row.")
    label_w, gap = 132, 28
    pw = (W - 2 * PAD - label_w - gap) / 2
    cw, ch = pw / 2, 84
    y = 104
    for panels in grid:
        cells_top = y + 58
        for i, (name, n) in enumerate((("Tortoise", n_pos), ("Not a tortoise", n_neg))):
            cy = cells_top + i * ch + ch / 2
            body.append(text(PAD, cy - 2, name, 13.5, t["ink"], 600))
            body.append(text(PAD, cy + 16, f"{n} images", 12.5, t["muted"]))
        for j, (name, sub, c, key) in enumerate(panels):
            x0 = PAD + label_w + j * (pw + gap)
            body.append(f'<circle cx="{x0 + 6}" cy="{y + 2}" r="5" fill="{t[key]}"/>')
            body.append(text(x0 + 18, y + 7, name, 14, t["ink"], 600))
            body.append(text(x0 + 18, y + 25, sub, 12.5, t["ink2"]))
            for k, col in enumerate(("Called tortoise", "Called not a tortoise")):
                body.append(text(x0 + k * cw + cw / 2, cells_top - 9, col, 12, t["muted"], anchor="middle"))
            for i, row in enumerate([[c["TP"], c["FN"]], [c["FP"], c["TN"]]]):
                tot = sum(row)
                for k, v in enumerate(row):
                    share = v / tot if tot else 0
                    fill = mix(t["surface"], t[key], 0.08 + 0.92 * share)
                    # labels inside a categorical fill are white; on a pale tint they take the ink colour
                    ink = "#ffffff" if share >= 0.5 else t["ink"]
                    x, yy = x0 + k * cw + 1, cells_top + i * ch + 1
                    body.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{cw - 2:.1f}" height="{ch - 2:.1f}" rx="4" fill="{fill}"/>')
                    body.append(text(x + (cw - 2) / 2, yy + ch / 2 + 1, f"{v}", 26, ink, 600, "middle"))
                    kind = ("found", "missed", "false alarm" if v == 1 else "false alarms", "correct")[i * 2 + k]
                    body.append(text(x + (cw - 2) / 2, yy + ch / 2 + 21, f"{kind} · {share:.0%}", 11.5, ink, 400,
                                     "middle", 'fill-opacity="0.88"'))
        y = cells_top + 2 * ch + 40
    h = y - 4
    body.append(text(PAD, h - 18, "Scored per image. One photo the manifest labels tortoise shows a bulldozer and a "
                     "truck; it is counted as not a tortoise.", 11.5, t["muted"]))
    flat = [p for row in grid for p in row]
    return svg(theme, h, body, "Confusion matrices: Gemini zero-shot vs Mask R-CNN",
               "; ".join(f"{p[0]}, {p[1]}: {p[2]['TP']} found, {p[2]['FN']} missed, {p[2]['FP']} false alarms, "
                         f"{p[2]['TN']} correct" for p in flat))


# ---------------------------------------------------------------- figure 2

SPECIES_NAMES = {"nine-banded armadillo": "Nine-banded armadillo", "feral hog": "Feral hog", "wild turkey": "Wild turkey",
                 "deer": "Deer", "bird": "Bird", "bobcat": "Bobcat"}
SERIES = ("gem", "neu", "mrc")


def false_alarm_groups(rows):
    groups: dict[str, list] = {}
    for r in rows:
        if r["truth"]:
            continue
        if r["file"] in LABEL_FIXES:
            key = "Bulldozer photo (labeled tortoise)"
        elif r["cls"] == "blank":
            key = f"Empty frame, {r['src']}"
        else:
            key = r["species"]
        g = groups.setdefault(key, {"n": 0, "gem": 0, "neu": 0, "mrc": 0})
        g["n"] += 1
        g["gem"] += r["gem"]
        g["neu"] += r["neu"]
        g["mrc"] += r["score"] > 0.5
    flagged = lambda g: any(g[s] for s in SERIES)  # noqa: E731
    folded = sorted(k for k, g in groups.items() if not flagged(g) and not k.startswith("Empty"))
    out = [dict(name=SPECIES_NAMES.get(k, k[0].upper() + k[1:]), detail="", **g)
           for k, g in groups.items() if k not in folded]
    out.append(dict(name="Other animals", detail=", ".join(folded),
                    **{f: sum(groups[k][f] for k in folded) for f in ("n",) + SERIES}))
    out.sort(key=lambda g: (-g["mrc"], -g["gem"], -g["n"]))
    return [g for g in out if flagged(g)] + [g for g in out if not flagged(g)]


def fig_false_alarms(theme, groups, totals):
    """Small multiples: one column per method, shared rows of image types."""
    t = THEMES[theme]
    n_neg = sum(g["n"] for g in groups)
    body = heading(t, "Where the false alarms come from",
                   f"The {n_neg} hold-out images without a tortoise, by what they show. "
                   "Bars count the images each method called tortoise.")
    label_w, gap, top, row_h, bar_h = 236, 26, 176, 40, 14
    cols = [("gem", "Gemini 3.8 Flash", "original prompt"), ("neu", "Gemini 3.8 Flash", "neutral prompt"),
            ("mrc", "Mask R-CNN", "threshold 0.5")]
    cw = (W - 2 * PAD - label_w - gap * (len(cols) - 1)) / len(cols)
    vmax = max(6, max(g["mrc"] for g in groups))
    bottom = top + row_h * len(groups)
    for i, g in enumerate(groups):
        y = top + i * row_h
        if i:
            body.append(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W - PAD}" y2="{y:.1f}" stroke="{t["grid"]}"/>')
        body.append(text(PAD, y + 18, g["name"], 13, t["ink"], 500))
        sub = f"{g['n']} images" + (f" · {g['detail']}" if g["detail"] else "")
        if len(sub) > 38:
            sub = f"{g['n']} images · " + ", ".join(g["detail"].split(", ")[:2]) + " and others"
        body.append(text(PAD, y + 33, sub, 11.5, t["muted"]))
    for j, (key, name, sub) in enumerate(cols):
        x0 = PAD + label_w + j * (cw + gap)
        x1 = x0 + cw - 30
        sx = lambda v: x0 + (x1 - x0) * v / vmax  # noqa: E731
        n_fp = totals[key]
        body.append(f'<circle cx="{x0 + 6}" cy="{102}" r="5" fill="{t[key]}"/>')
        body.append(text(x0 + 18, 107, name, 13.5, t["ink"], 600))
        body.append(text(x0 + 18, 125, sub, 12.5, t["ink2"]))
        body.append(text(x0 + 18, 146, f"{n_fp} false alarm{'' if n_fp == 1 else 's'} in total", 12.5, t["ink"], 600))
        for v in range(0, vmax + 1, 3):
            body.append(text(sx(v), bottom + 18, str(v), 11.5, t["muted"], anchor="middle"))
        body.append(f'<line x1="{x0}" y1="{top - 8}" x2="{x0}" y2="{bottom}" stroke="{t["axis"]}"/>')
        for i, g in enumerate(groups):
            v = g[key]
            by = top + i * row_h + (row_h - bar_h) / 2
            body.append(hbar(x0, by, sx(v) - x0, bar_h, t[key]))
            body.append(text(sx(v) + 6, by + bar_h - 2.5, str(v), 12, t["ink"] if v else t["muted"], 600 if v else 400))
    body.append(text(W - PAD, bottom + 40, "Images called tortoise, per method", 12, t["muted"], anchor="end"))
    h = bottom + 58
    return svg(theme, h, body, "False alarms by image type",
               "; ".join(f"{g['name']} ({g['n']} images): Gemini original prompt {g['gem']}, neutral prompt {g['neu']}, "
                         f"Mask R-CNN {g['mrc']}" for g in groups))


# ---------------------------------------------------------------- figure 3

def fig_tradeoff(theme, rows, gem_c, neu_c, n_pos, n_neg):
    t = THEMES[theme]
    body = heading(t, "No Mask R-CNN threshold reaches Gemini's corner",
                   "Up and to the left is better. The orange line is Mask R-CNN at every confidence threshold; "
                   "each Gemini prompt is one point.")
    body += legend(t, [("neu", "Gemini, neutral prompt", "dot"), ("gem", "Gemini, original prompt", "dot"),
                       ("mrc", "Mask R-CNN as its threshold varies", "line")], 100, step=230)

    left, right, top, bottom = PAD + 52, W - PAD - 20, 130, 470
    xmax, ymin, ymax = 25, 138, 168
    sx = lambda v: left + (right - left) * v / xmax  # noqa: E731
    sy = lambda v: bottom - (bottom - top) * (v - ymin) / (ymax - ymin)  # noqa: E731
    for v in range(0, xmax + 1, 5):
        body.append(f'<line x1="{sx(v):.1f}" y1="{top}" x2="{sx(v):.1f}" y2="{bottom}" stroke="{t["grid"]}"/>')
        body.append(text(sx(v), bottom + 18, str(v), 12, t["muted"], anchor="middle"))
    for v in (140, 145, 150, 155, 160):
        body.append(f'<line x1="{left}" y1="{sy(v):.1f}" x2="{right}" y2="{sy(v):.1f}" stroke="{t["grid"]}"/>')
        body.append(text(left - 8, sy(v) + 4, str(v), 12, t["muted"], anchor="end"))
    body.append(f'<line x1="{left}" y1="{sy(n_pos):.1f}" x2="{right}" y2="{sy(n_pos):.1f}" stroke="{t["axis"]}"/>')
    body.append(text(left - 8, sy(n_pos) + 4, f"{n_pos}", 12, t["ink2"], 600, "end"))
    body.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{t["axis"]}"/>')
    body.append(text((left + right) / 2, bottom + 40, f"False alarms (of {n_neg} images without a tortoise)",
                     12.5, t["muted"], anchor="middle"))
    body.append(text(PAD + 4, (top + bottom) / 2, f"Tortoises found (of {n_pos})", 12.5, t["muted"], anchor="middle",
                     extra=f'transform="rotate(-90 {PAD + 4} {(top + bottom) / 2})"'))

    # ROC staircase: lower the threshold past each image's score in turn
    pts, tp, fp = [(0, 0)], 0, 0
    for r in sorted(rows, key=lambda r: -r["score"]):
        tp, fp = tp + r["truth"], fp + (1 - r["truth"])
        pts.append((fp, tp))
    path = []
    for (fa, ta), (fb, tb) in zip(pts, pts[1:]):
        path += [(fb, ta), (fb, tb)]
    d = "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in [pts[0]] + path)
    body.append(f'<clipPath id="plot"><rect x="{left}" y="{top - 6}" width="{right - left}" height="{bottom - top + 6}"/></clipPath>')
    body.append(f'<path d="{d}" fill="none" stroke="{t["mrc"]}" stroke-width="2" stroke-linejoin="round" '
                f'stroke-linecap="round" clip-path="url(#plot)"/>')
    beyond = sum(1 - r["truth"] for r in rows if r["score"] > 0.05)
    body.append(text(right, sy(n_pos) - 10, f"continues to {beyond} false alarms at threshold 0.05 →", 11.5,
                     t["muted"], anchor="end"))

    def mark(thr, dx, dy, anchor="start"):
        tp_ = sum(r["truth"] for r in rows if r["score"] > thr)
        fp_ = sum(1 - r["truth"] for r in rows if r["score"] > thr)
        cx, cy = sx(fp_), sy(tp_)
        body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{t["mrc"]}" stroke="{t["surface"]}" stroke-width="2"/>')
        body.append(text(cx + dx, cy + dy, f"threshold {thr}", 12, t["ink"], 500, anchor))
        body.append(text(cx + dx, cy + dy + 15, f"{fp_} false alarms, {n_pos - tp_} missed", 11.5, t["ink2"], 400, anchor))
    mark(0.5, 0, 24, "middle")
    mark(0.85, 8, 24)
    mark(0.95, 12, 30)
    mark(0.99, 10, 24)
    for c, key in ((gem_c, "gem"), (neu_c, "neu")):
        body.append(f'<circle cx="{sx(c["FP"]):.1f}" cy="{sy(c["TP"]):.1f}" r="7" fill="{t[key]}" '
                    f'stroke="{t["surface"]}" stroke-width="2"/>')
    gy = sy(n_pos)
    body.append(text(left + 2, gy - 32, "Gemini, zero-shot", 12.5, t["ink"], 600))
    body.append(text(left + 2, gy - 16, f"neutral prompt: {neu_c['FP']} false alarms  ·  original prompt: {gem_c['FP']}  ·  "
                     f"none missed", 11.5, t["ink2"]))
    h = bottom + 64
    return svg(theme, h, body, "Threshold trade-off: Mask R-CNN vs Gemini",
               f"Gemini neutral prompt: {neu_c['FP']} false alarms, {neu_c['TP']} of {n_pos} found. Original prompt: "
               f"{gem_c['FP']} false alarms, {gem_c['TP']} found. Mask R-CNN staircase over all thresholds.")


# ---------------------------------------------------------------- figure 4

SPECIES_ROWS = [  # (group, label, note)
    ("Texas tortoise", "Texas tortoise", "the right species"),
    ("tortoise", "“Tortoise”", "no species given"),
    ("desert tortoise", "Desert tortoise", "same genus, Mojave and Sonoran deserts"),
    ("gopher tortoise", "Gopher tortoise", "same genus, southeastern US"),
    ("red-footed tortoise", "Red-footed tortoise", "South American"),
    ("other tortoise", "Other tortoise species", ""),
    ("turtle", "Box turtle", "a turtle; still counted as found"),
]


def fig_species(theme, counts, others):
    t = THEMES[theme]
    n = sum(counts["gem"].values())
    body = heading(t, "Without the hint, Gemini still finds every tortoise but often names the wrong one",
                   f"What Gemini called the {n} tortoise photos, all of them Texas tortoises. "
                   "Every answer below counts as found.")
    body += legend(t, [("gem", "Original prompt, lists “Texas tortoise” as an example", "square"),
                       ("neu", "Neutral prompt, names no animals", "square")], 100, step=400)
    label_w, top, row_h, bar_h = 262, 130, 50, 12
    x0, x1 = PAD + label_w, W - PAD - 76
    vmax = 160
    sx = lambda v: x0 + (x1 - x0) * v / vmax  # noqa: E731
    bottom = top + row_h * len(SPECIES_ROWS)
    for v in range(0, vmax + 1, 40):
        body.append(f'<line x1="{sx(v):.1f}" y1="{top - 6}" x2="{sx(v):.1f}" y2="{bottom}" stroke="{t["grid"]}"/>')
        body.append(text(sx(v), bottom + 18, str(v), 12, t["muted"], anchor="middle"))
    body.append(text((x0 + x1) / 2, bottom + 38, "Tortoise photos", 12, t["muted"], anchor="middle"))
    body.append(f'<line x1="{x0}" y1="{top - 6}" x2="{x0}" y2="{bottom}" stroke="{t["axis"]}"/>')
    for i, (grp, label, note) in enumerate(SPECIES_ROWS):
        y = top + i * row_h
        body.append(text(PAD, y + 17, label, 13, t["ink"], 600 if i == 0 else 500))
        if grp == "other tortoise":
            note = ", ".join(others[:3]) + (f" and {len(others) - 3} more" if len(others) > 3 else "")
        body.append(text(PAD, y + 33, note, 11.5, t["muted"]))
        for k, s in enumerate(("gem", "neu")):
            v = counts[s].get(grp, 0)
            by = y + 9 + k * (bar_h + 2)
            body.append(hbar(x0, by, sx(v) - x0, bar_h, t[s]))
            lab = f"{v}  ({v / n:.0%})" if i == 0 else str(v)
            body.append(text(sx(v) + 6, by + bar_h - 2, lab, 11.5, t["ink2"] if v else t["muted"],
                             600 if i == 0 and v else (500 if v else 400)))
    h = bottom + 60
    return svg(theme, h, body, "Species Gemini named for the tortoise photos, by prompt",
               "; ".join(f"{label}: original prompt {counts['gem'].get(g, 0)}, neutral prompt {counts['neu'].get(g, 0)}"
                         for g, label, _ in SPECIES_ROWS))


# ---------------------------------------------------------------- main

def main():
    rows = load()
    n_pos = sum(r["truth"] for r in rows)
    n_neg = len(rows) - n_pos
    c = {"gem": confusion(rows, lambda r: r["gem"]), "neu": confusion(rows, lambda r: r["neu"]),
         "m50": confusion(rows, lambda r: r["score"] > 0.5), "m85": confusion(rows, lambda r: r["score"] > 0.85),
         "m95": confusion(rows, lambda r: r["score"] > 0.95)}

    grid = [[("Gemini 3.8 Flash", "original prompt, lists example animals", c["gem"], "gem"),
             ("Gemini 3.8 Flash", "neutral prompt, names no animals", c["neu"], "neu")],
            [("Mask R-CNN", "threshold 0.5, the paper's default", c["m50"], "mrc"),
             ("Mask R-CNN", "threshold 0.85, the paper's best F1", c["m85"], "mrc")]]
    groups = false_alarm_groups(rows)
    tort = [r for r in rows if r["truth"]]
    counts = {s: collections.Counter(species_group(r[s + "_name"]) for r in tort) for s in ("gem", "neu")}
    seen: dict[str, str] = {}  # lower-case name -> spelling as Gemini wrote it
    for s in ("neu", "gem"):
        for r in tort:
            if species_group(r[s + "_name"]) == "other tortoise":
                seen.setdefault(r[s + "_name"].lower(), r[s + "_name"])
    others = sorted(seen, key=lambda x: (-sum(r["neu_name"].lower() == x for r in tort), x))
    others = [seen[o].replace(" tortoise", "") for o in others]
    others[:1] = [o[0].upper() + o[1:] for o in others[:1]]

    os.makedirs(FIGURES, exist_ok=True)
    totals = {"gem": c["gem"]["FP"], "neu": c["neu"]["FP"], "mrc": c["m50"]["FP"]}
    for theme in THEMES:
        for name, content in (("confusion", fig_confusion(theme, grid, n_pos, n_neg)),
                              ("false-alarms", fig_false_alarms(theme, groups, totals)),
                              ("tradeoff", fig_tradeoff(theme, rows, c["gem"], c["neu"], n_pos, n_neg)),
                              ("species", fig_species(theme, counts, others))):
            with open(os.path.join(FIGURES, f"{name}-{theme}.svg"), "w", encoding="utf-8", newline="\n") as f:
                f.write(content)

    print(f"{n_pos} tortoise, {n_neg} not\n")
    for name, k in (("Gemini original", "gem"), ("Gemini neutral", "neu"), ("Mask R-CNN 0.5", "m50"),
                    ("Mask R-CNN 0.85", "m85"), ("Mask R-CNN 0.95", "m95")):
        x = c[k]
        p = x["TP"] / (x["TP"] + x["FP"])
        rec = x["TP"] / (x["TP"] + x["FN"])
        lo, hi = clopper_pearson(x["FP"], n_neg)
        print(f"{name:16} {x}  precision {p:.3f}  recall {rec:.3f}  F1 {2 * p * rec / (p + rec):.3f}  "
              f"accuracy {(x['TP'] + x['TN']) / len(rows):.3f}  FPR {x['FP'] / n_neg:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    neg = [r for r in rows if not r["truth"]]
    for g in ("gem", "neu"):
        b = sum(1 for r in neg if r["score"] > 0.5 and not r[g])
        cc = sum(1 for r in neg if r[g] and not r["score"] > 0.5)
        print(f"McNemar vs Mask R-CNN 0.5 ({g}): Mask-only {b}, Gemini-only {cc}, p = {mcnemar_exact(b, cc):.2g}")
    print("\nfalse-alarm groups:", groups)
    print("species counts:", {s: dict(v) for s, v in counts.items()}, "| other species:", others)
    print(f"\nfigures written to {FIGURES}")


if __name__ == "__main__":
    main()
