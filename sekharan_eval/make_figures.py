"""Draw the README figures from the prediction files in results/.

Writes three charts, each as a light and a dark SVG, to figures/:
    confusion-{light,dark}.svg     confusion matrices for Gemini and Mask R-CNN
    false-alarms-{light,dark}.svg  which non-tortoise images each model called tortoise
    tradeoff-{light,dark}.svg      Mask R-CNN's threshold curve against Gemini's point
and prints the numbers the README quotes (counts, rates with 95% Clopper-Pearson
intervals, and an exact McNemar test on the non-tortoise images).

Standard library only. Usage:
    python make_figures.py
"""
from __future__ import annotations

import csv
import math
import os
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

# The manifest labels this photo a tortoise, but it shows a bulldozer and a
# pickup truck and no tortoise, so it is scored as a negative here.
LABEL_FIXES = {"test/tortoise/inat-tortoise_inat50015197_p79414426.jpg": 0}

THEMES = {
    "light": dict(surface="#fcfcfb", ring="#0b0b0b", ring_op=0.10, ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7", gem="#2a78d6", mrc="#eb6834"),
    "dark": dict(surface="#1a1a19", ring="#ffffff", ring_op=0.10, ink="#f0efec", ink2="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835", gem="#3987e5", mrc="#d95926"),
}
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
W = 900
PAD = 32


# ---------------------------------------------------------------- data

def load() -> list[dict]:
    gem = {r["file"]: r for r in csv.DictReader(open(os.path.join(RESULTS, "predictions_gemini-3.8-flash.csv"), encoding="utf-8"))}
    rows = []
    for r in csv.DictReader(open(os.path.join(RESULTS, "predictions_maskrcnn.csv"), encoding="utf-8")):
        f = r["file"]
        src = "South Texas" if "Kutugata" in r["source"] else ("Florida" if "NACTI" in r["source"] else "iNaturalist")
        rows.append({"file": f, "truth": LABEL_FIXES.get(f, int(r["truth_positive"])),
                     "cls": r["truth_class"], "species": r["truth_species"], "src": src,
                     "score": float(r["max_score"]), "gem": int(gem[f]["pred_positive"])})
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


# ---------------------------------------------------------------- figure 1

def fig_confusion(theme, panels, n_pos, n_neg):
    t = THEMES[theme]
    body = heading(t, f"Both find all {n_pos} tortoises. Mask R-CNN also calls {panels[1][2]['FP']} other images tortoise; Gemini calls 1.",
                   f"Hold-out set: {n_pos} tortoise photos and {n_neg} images without a tortoise. "
                   "Shade shows each cell's share of its row.")
    label_w, gap = 136, 26
    pw = (W - 2 * PAD - label_w - gap * (len(panels) - 1)) / len(panels)
    cw, ch, top = pw / 2, 92, 150
    # row labels
    for i, (name, n) in enumerate((("Tortoise", n_pos), ("Not a tortoise", n_neg))):
        cy = top + i * ch + ch / 2
        body.append(text(PAD, cy - 2, name, 13.5, t["ink"], 600))
        body.append(text(PAD, cy + 16, f"{n} images", 12.5, t["muted"]))
    for j, (name, sub, c, color) in enumerate(panels):
        x0 = PAD + label_w + j * (pw + gap)
        body.append(f'<circle cx="{x0 + 6}" cy="{top - 58}" r="5" fill="{t[color]}"/>')
        body.append(text(x0 + 18, top - 53, name, 14, t["ink"], 600))
        body.append(text(x0 + 18, top - 35, sub, 12.5, t["ink2"]))
        for k, col in enumerate(("Called tortoise", "Called not")):
            body.append(text(x0 + k * cw + cw / 2, top - 10, col, 12, t["muted"], anchor="middle"))
        cells = [[c["TP"], c["FN"]], [c["FP"], c["TN"]]]
        for i, row in enumerate(cells):
            tot = sum(row)
            for k, v in enumerate(row):
                share = v / tot if tot else 0
                fill = mix(t["surface"], t[color], 0.08 + 0.92 * share)
                # labels inside a categorical fill are white; on a pale tint they take the ink colour
                ink = "#ffffff" if share >= 0.5 else t["ink"]
                x, y = x0 + k * cw + 1, top + i * ch + 1
                body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw - 2:.1f}" height="{ch - 2:.1f}" rx="4" fill="{fill}"/>')
                body.append(text(x + (cw - 2) / 2, y + ch / 2 + 2, f"{v}", 26, ink, 600, "middle"))
                kind = ("found", "missed", "false alarm" if v == 1 else "false alarms", "correct")[i * 2 + k]
                body.append(text(x + (cw - 2) / 2, y + ch / 2 + 22, f"{kind} · {share:.0%}", 11.5, ink, 400, "middle",
                                 'fill-opacity="0.85"'))
    h = top + 2 * ch + 36
    body.append(text(PAD, h - 16, "Scored per image. One photo the manifest labels tortoise shows a bulldozer and a truck; "
                     "it is counted as not a tortoise.", 11.5, t["muted"]))
    return svg(theme, h, body, "Confusion matrices: Gemini zero-shot vs Mask R-CNN",
               "; ".join(f"{p[0]} {p[1]}: TP {p[2]['TP']}, FN {p[2]['FN']}, FP {p[2]['FP']}, TN {p[2]['TN']}" for p in panels))


# ---------------------------------------------------------------- figure 2

SPECIES_NAMES = {"nine-banded armadillo": "Nine-banded armadillo", "feral hog": "Feral hog", "wild turkey": "Wild turkey",
                 "deer": "Deer", "bird": "Bird", "bobcat": "Bobcat"}


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
        g = groups.setdefault(key, [0, 0, 0])
        g[0] += 1
        g[1] += r["gem"]
        g[2] += r["score"] > 0.5
    folded = [k for k, v in groups.items() if v[1] == v[2] == 0 and not k.startswith("Empty")]
    other = [sum(groups[k][i] for k in folded) for i in range(3)]
    out = [(SPECIES_NAMES.get(k, k[0].upper() + k[1:]), "", *v) for k, v in groups.items() if k not in folded]
    out.append(("Other animals", ", ".join(sorted(folded)), *other))
    out.sort(key=lambda g: (-g[4], -g[3] if not g[0].startswith(("Other", "Empty frame, F")) else 0, g[0]))
    # keep the two all-zero rows last
    return [g for g in out if g[4] or g[3]] + [g for g in out if not (g[4] or g[3])]


def fig_false_alarms(theme, groups, gem_fp, mrc_fp):
    t = THEMES[theme]
    body = heading(t, "Where the false alarms come from",
                   f"The {sum(g[2] for g in groups)} hold-out images without a tortoise, by what they show. "
                   "Bars count the images each model called tortoise.")
    # legend
    lx, ly = PAD, 100
    for color, label in ((t["gem"], f"Gemini 3.8 Flash, zero-shot  ({gem_fp} in total)"),
                         (t["mrc"], f"Mask R-CNN, threshold 0.5  ({mrc_fp} in total)")):
        body.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="3" fill="{color}"/>')
        body.append(text(lx + 20, ly, label, 12.5, t["ink2"]))
        lx += 300
    label_w, top, row_h, bar_h = 250, 128, 44, 11
    x0, x1 = PAD + label_w, W - PAD - 40
    vmax = max(6, max(g[4] for g in groups))
    sx = lambda v: x0 + (x1 - x0) * v / vmax  # noqa: E731
    bottom = top + row_h * len(groups)
    for v in range(0, vmax + 1, 2):
        body.append(f'<line x1="{sx(v):.1f}" y1="{top - 6}" x2="{sx(v):.1f}" y2="{bottom}" stroke="{t["grid"]}" stroke-width="1"/>')
        body.append(text(sx(v), bottom + 18, str(v), 12, t["muted"], anchor="middle", extra='font-variant-numeric="tabular-nums"'))
    body.append(text((x0 + x1) / 2, bottom + 38, "Images called tortoise", 12, t["muted"], anchor="middle"))
    body.append(f'<line x1="{x0}" y1="{top - 6}" x2="{x0}" y2="{bottom}" stroke="{t["axis"]}" stroke-width="1"/>')
    for i, (name, detail, n, g, m) in enumerate(groups):
        y = top + i * row_h
        body.append(text(PAD, y + 15, name, 13, t["ink"], 500))
        sub = f"{n} images" + (f" · {detail}" if detail else "")
        body.append(text(PAD, y + 31, sub if len(sub) < 40 else f"{n} images · " + detail.split(",")[0] + " and others",
                         11.5, t["muted"]))
        for k, (v, color) in enumerate(((g, t["gem"]), (m, t["mrc"]))):
            by = y + 6 + k * (bar_h + 2)
            body.append(hbar(x0, by, sx(v) - x0, bar_h, color))
            body.append(text(sx(v) + 6, by + bar_h - 1.5, str(v), 11.5, t["ink2"] if v else t["muted"], 500 if v else 400))
    h = bottom + 60
    return svg(theme, h, body, "False alarms by image type",
               "; ".join(f"{g[0]} ({g[2]} images): Gemini {g[3]}, Mask R-CNN {g[4]}" for g in groups))


# ---------------------------------------------------------------- figure 3

def fig_tradeoff(theme, rows, gem_c, n_pos, n_neg):
    t = THEMES[theme]
    body = heading(t, "No Mask R-CNN threshold reaches Gemini's corner",
                   "Up and to the left is better. The orange line is Mask R-CNN at every confidence threshold; "
                   "Gemini has a single operating point.")
    # legend
    ly = 100
    body.append(f'<circle cx="{PAD + 6}" cy="{ly - 4}" r="5" fill="{t["gem"]}"/>')
    body.append(text(PAD + 18, ly, "Gemini 3.8 Flash, zero-shot", 12.5, t["ink2"]))
    body.append(f'<line x1="{PAD + 250}" y1="{ly - 4}" x2="{PAD + 270}" y2="{ly - 4}" stroke="{t["mrc"]}" stroke-width="2" stroke-linecap="round"/>')
    body.append(text(PAD + 278, ly, "Mask R-CNN as its threshold varies", 12.5, t["ink2"]))

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
    body.append(text((left + right) / 2, bottom + 40, f"False alarms (of {n_neg} images without a tortoise)", 12.5, t["muted"], anchor="middle"))
    body.append(text(PAD + 4, (top + bottom) / 2, f"Tortoises found (of {n_pos})", 12.5, t["muted"], anchor="middle",
                     extra=f'transform="rotate(-90 {PAD + 4} {(top + bottom) / 2})"'))

    # ROC staircase: lower the threshold past each image's score in turn
    pts, tp, fp = [(0, 0)], 0, 0
    for r in sorted(rows, key=lambda r: -r["score"]):
        tp, fp = tp + r["truth"], fp + (1 - r["truth"])
        pts.append((fp, tp))
    path = []
    for (fa, ta), (fb, tb) in zip(pts, pts[1:]):
        path.append((fb, ta))
        path.append((fb, tb))
    d = "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in [pts[0]] + path)
    body.append(f'<clipPath id="plot"><rect x="{left}" y="{top - 6}" width="{right - left}" height="{bottom - top + 6}"/></clipPath>')
    body.append(f'<path d="{d}" fill="none" stroke="{t["mrc"]}" stroke-width="2" stroke-linejoin="round" '
                f'stroke-linecap="round" clip-path="url(#plot)"/>')
    beyond = sum(1 - r["truth"] for r in rows if r["score"] > 0.05)
    body.append(text(right, sy(n_pos) - 10, f"continues to {beyond} false alarms at threshold 0.05 →", 11.5, t["muted"], anchor="end"))

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
    gx, gy = sx(gem_c["FP"]), sy(gem_c["TP"])
    body.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="7" fill="{t["gem"]}" stroke="{t["surface"]}" stroke-width="2"/>')
    body.append(text(gx - 6, gy - 32, "Gemini, zero-shot", 12.5, t["ink"], 600))
    body.append(text(gx - 6, gy - 16, f"{gem_c['FP']} false alarm, {gem_c['FN']} missed", 11.5, t["ink2"]))
    h = bottom + 64
    return svg(theme, h, body, "Threshold trade-off: Mask R-CNN vs Gemini",
               f"Gemini: {gem_c['FP']} false alarms, {gem_c['TP']} of {n_pos} found. Mask R-CNN staircase over all thresholds.")


# ---------------------------------------------------------------- main

def main():
    rows = load()
    n_pos = sum(r["truth"] for r in rows)
    n_neg = len(rows) - n_pos
    gem = confusion(rows, lambda r: r["gem"])
    m50 = confusion(rows, lambda r: r["score"] > 0.5)
    m85 = confusion(rows, lambda r: r["score"] > 0.85)
    m95 = confusion(rows, lambda r: r["score"] > 0.95)

    os.makedirs(FIGURES, exist_ok=True)
    panels = [("Gemini 3.8 Flash", "zero-shot, no training", gem, "gem"),
              ("Mask R-CNN", "threshold 0.5 (paper's default)", m50, "mrc"),
              ("Mask R-CNN", "threshold 0.85 (paper's best F1)", m85, "mrc")]
    groups = false_alarm_groups(rows)
    for theme in THEMES:
        for name, content in (("confusion", fig_confusion(theme, panels, n_pos, n_neg)),
                              ("false-alarms", fig_false_alarms(theme, groups, gem["FP"], m50["FP"])),
                              ("tradeoff", fig_tradeoff(theme, rows, gem, n_pos, n_neg))):
            with open(os.path.join(FIGURES, f"{name}-{theme}.svg"), "w", encoding="utf-8", newline="\n") as f:
                f.write(content)

    print(f"{n_pos} tortoise, {n_neg} not\n")
    for name, c in (("Gemini", gem), ("Mask R-CNN 0.5", m50), ("Mask R-CNN 0.85", m85), ("Mask R-CNN 0.95", m95)):
        p = c["TP"] / (c["TP"] + c["FP"])
        rec = c["TP"] / (c["TP"] + c["FN"])
        lo, hi = clopper_pearson(c["FP"], n_neg)
        print(f"{name:16} {c}  precision {p:.3f}  recall {rec:.3f}  F1 {2 * p * rec / (p + rec):.3f}  "
              f"accuracy {(c['TP'] + c['TN']) / len(rows):.3f}  FPR {c['FP'] / n_neg:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    b = sum(1 for r in rows if not r["truth"] and r["score"] > 0.5 and not r["gem"])
    c = sum(1 for r in rows if not r["truth"] and r["gem"] and not r["score"] > 0.5)
    print(f"\nnon-tortoise images: Mask R-CNN-only false alarms {b}, Gemini-only {c}, "
          f"exact McNemar p = {mcnemar_exact(b, c):.2g}")
    print("false-alarm groups:", groups)
    print(f"\nfigures written to {FIGURES}")


if __name__ == "__main__":
    main()
