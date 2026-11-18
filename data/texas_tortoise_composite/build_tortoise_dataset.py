"""Build a public-image stand-in for the Texas tortoise image sets in
Tleimat, Krell & McCracken (2026), "Needle in a haystack", Amphibian & Reptile
Conservation 20(1): 67-81, Table 1: training set M4 (2,705 images) and the
hold-out test set (274 images).

The counts per split and class match Table 1 exactly. The images are not the
paper's (those were never published); they come from three public sources:
  * iNaturalist research-grade photos with Creative Commons licences
  * NACTI (North American Camera Trap Images, LILA BC), Archbold FL ranch site
  * github.com/mkutu/Nilgai (MIT), South Texas trail-camera sample frames

Hold-out images never share a photographer (iNaturalist) or a camera (NACTI)
with training images.

Usage:  python -I build_tortoise_dataset.py OUT_DIR CACHE_DIR
Standard library only. Re-running with the same CACHE_DIR reproduces the same
selection (fixed seed, cached API responses) and skips files already present.
"""
import collections
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

OUT, CACHE = sys.argv[1], sys.argv[2]
SEED = 20261116
UA = "dataset-compiler/1.0"
# No-derivatives licences are left out so the images can be resized/augmented.
INAT_LIC = ["cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"]
STUDY_BOX = dict(nelat=28.6, nelng=-97.0, swlat=25.8, swlng=-99.8)  # the paper's four sites
NACTI_BASE = "https://storage.googleapis.com/public-datasets-lila/nacti-unzipped/"
NACTI_CSV = "https://storage.googleapis.com/public-datasets-lila/nacti/nacti_metadata.csv.zip"
NILGAI_RAW = "https://raw.githubusercontent.com/mkutu/Nilgai/HEAD/"

NACTI_SPECIES = {  # NACTI common name -> label used here (all occur in South Texas)
    "nine-banded armadillo": "nine-banded armadillo",
    "raccoon": "raccoon", "wild boar": "feral hog", "unidentified deer": "deer",
    "coyote": "coyote", "bobcat": "bobcat", "domestic cow": "cattle",
    "unidentified bird": "bird", "virginia opossum": "opossum",
    "unidentified rabbit": "rabbit", "wild turkey": "wild turkey",
}

NILGAI_NAMES = {"Birds": "bird", "White-Tailed Deer": "deer", "Pig": "feral hog"}  # same names as NACTI

# stratum -> (class, train, test).  Totals per class/split equal the paper's Table 1.
PLAN = {
    "inat_tortoise":            ("tortoise",     1332, 165),
    "nilgai_blank":             ("blank",         135,  21),
    "nacti_blank":              ("blank",         285,  44),
    "inat_armadillo":           ("other_animal",  167,   8),
    "nacti_nine-banded armadillo": ("other_animal", 167, 8),
    "nilgai_animals":           ("other_animal",   62,   3),
    "nacti_raccoon":            ("other_animal",   61,   3),
    "nacti_feral hog":          ("other_animal",   61,   3),
    "nacti_deer":               ("other_animal",   61,   3),
    "nacti_coyote":             ("other_animal",   61,   3),
    "nacti_bobcat":             ("other_animal",   61,   3),
    "nacti_cattle":             ("other_animal",   62,   2),
    "nacti_bird":               ("other_animal",   62,   2),
    "nacti_opossum":            ("other_animal",   48,   2),
    "nacti_rabbit":             ("other_animal",   42,   2),
    "nacti_wild turkey":        ("other_animal",   38,   2),
}
PAPER = {("train", "tortoise"): 1332, ("train", "blank"): 420, ("train", "other_animal"): 953,
         ("test", "tortoise"): 165, ("test", "blank"): 65, ("test", "other_animal"): 44}

rng = random.Random(SEED)
os.makedirs(CACHE, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fetch(url, tries=5, timeout=90):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    raise last


def cached(url, name):
    p = os.path.join(CACHE, name)
    if not os.path.exists(p):
        data = fetch(url)
        with open(p + ".part", "wb") as f:
            f.write(data)
        os.replace(p + ".part", p)
    return p


# ---------------------------------------------------------------- iNaturalist
def inat_all(tag, taxon, extra):
    p = os.path.join(CACHE, f"inat_{tag}.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    params = dict(taxon_name=taxon, photos="true", quality_grade="research", captive="false",
                  photo_license=",".join(INAT_LIC), per_page=200, order_by="id", order="asc", **extra)
    out, id_above = [], 0
    while True:
        params["id_above"] = id_above
        d = json.loads(fetch("https://api.inaturalist.org/v1/observations?" + urllib.parse.urlencode(params)))
        res = d["results"]
        if not res:
            break
        out += res
        id_above = res[-1]["id"]
        log(f"iNat {tag}: {len(out)} / {d['total_results']}")
        time.sleep(1.2)
    json.dump(out, open(p, "w", encoding="utf-8"))
    return out


def ann_pairs(o):
    pairs = set()
    for a in o.get("annotations") or []:
        attr = a.get("controlled_attribute_id") or (a.get("controlled_attribute") or {}).get("id")
        val = a.get("controlled_value_id") or (a.get("controlled_value") or {}).get("id")
        pairs.add((attr, val))
    return pairs


def inat_candidates(obs, stratum, species):
    cands, dropped = [], collections.Counter()
    for o in obs:
        pairs = ann_pairs(o)
        if (17, 19) in pairs:                       # annotated Dead
            dropped["dead"] += 1
            continue
        if any(a == 22 and v != 24 for a, v in pairs):  # evidence other than the organism (bone, track...)
            dropped["not organism"] += 1
            continue
        ph = next((p for p in o.get("photos") or [] if (p.get("license_code") or "") in INAT_LIC), None)
        if not ph or "/square." not in (ph.get("url") or ""):
            dropped["no usable photo"] += 1
            continue
        cands.append(dict(
            stratum=stratum, species=species, source="iNaturalist", camera_trap="no",
            group=f"inat_user_{o['user']['id']}",
            source_id=f"inat{o['id']}_p{ph['id']}",
            image_url=ph["url"].replace("/square.", "/large."),
            source_page=o.get("uri") or f"https://www.inaturalist.org/observations/{o['id']}",
            license=ph["license_code"].upper(), attribution=ph.get("attribution") or "",
            location=(o.get("place_guess") or "").replace("\n", " "),
            latlon=o.get("location") or "", date=o.get("observed_on") or ""))
    log(f"{stratum}: {len(cands)} candidates, dropped {dict(dropped)}")
    return cands


# ---------------------------------------------------------------- NACTI (Florida)
def nacti_candidates():
    z = zipfile.ZipFile(cached(NACTI_CSV, "nacti_metadata.csv.zip"))
    rows = csv.DictReader(io.TextIOWrapper(z.open("nacti_metadata.csv"), encoding="utf-8"))
    events = collections.defaultdict(list)   # (stratum, event key) -> rows
    for row in rows:
        fid = row["id"]
        if not fid.startswith("FL-"):
            continue
        if row["name"] == "empty" or row["common_name"] == "empty":
            m = re.match(r"(FL-\d+)_(\d+)", fid)
            if m:
                cam, ctr = m.group(1), int(m.group(2))
                events[("nacti_blank", cam, "", ctr // 30)].append(row)
        elif row["common_name"] in NACTI_SPECIES:
            m = re.match(r"(FL-\d+)_(\d\d_\d\d_\d{4})_FL-\d+_(\d+)", fid)
            if m:
                cam, day, ctr = m.group(1), m.group(2), int(m.group(3))
                lab = NACTI_SPECIES[row["common_name"]]
                events[(f"nacti_{lab}", cam, day, ctr // 30)].append(row)
    # one image per trigger event, at most 3 events per camera-day
    per_day = collections.Counter()
    out = collections.defaultdict(list)
    keys = sorted(events)
    rng.shuffle(keys)
    for k in keys:
        stratum, cam, day, _ = k
        if day and per_day[(stratum, cam, day)] >= 3:
            continue
        per_day[(stratum, cam, day)] += 1
        row = rng.choice(sorted(events[k], key=lambda r: r["id"]))
        lab = "none" if stratum == "nacti_blank" else stratum[len("nacti_"):]
        out[stratum].append(dict(
            stratum=stratum, species=lab, source="NACTI (LILA BC), Archbold FL", camera_trap="yes",
            group=f"nacti_{cam}", source_id=os.path.splitext(row["id"])[0],
            image_url=NACTI_BASE + row["filename"],
            source_page="https://lila.science/datasets/nacti/",
            license="CDLA-Permissive-1.0", attribution="Tabak et al. 2019, USDA NWRC; via LILA BC",
            location="Archbold, Florida, USA",
            latlon="", date=day.replace("_", "/") if day else ""))
    for s, L in sorted(out.items()):
        log(f"{s}: {len(L)} candidate events on {len({c['group'] for c in L})} cameras")
    return out


# ---------------------------------------------------------------- Nilgai repo (South Texas)
def nilgai_candidates():
    p = cached(NILGAI_RAW + "notebooks/image_labels.csv", "nilgai_image_labels.csv")
    blank, animals = [], []
    for row in csv.DictReader(open(p, encoding="utf-8")):
        lab = row["MULTI_LABEL"]
        if lab == "Humans":
            continue
        is_blank = lab == "None"
        c = dict(stratum="nilgai_blank" if is_blank else "nilgai_animals",
                 species="none" if is_blank else NILGAI_NAMES.get(lab, lab.lower()),
                 source="Kutugata et al. 2021 sample (github.com/mkutu/Nilgai), South Texas", camera_trap="yes",
                 group=f"nilgai_{row['IMGNAME']}", source_id="nilgai_" + os.path.splitext(row["IMGNAME"])[0],
                 image_url=NILGAI_RAW + "images/images/" + row["IMGNAME"],
                 source_page="https://github.com/mkutu/Nilgai", license="MIT",
                 attribution="Copyright (c) 2020 Matthew Kutugata", location="South Texas, USA",
                 latlon="", date="")
        (blank if is_blank else animals).append(c)
    log(f"nilgai: {len(blank)} blank, {len(animals)} animal frames")
    return {"nilgai_blank": blank, "nilgai_animals": animals}


# ---------------------------------------------------------------- grouped split
def split_source(cands_by_stratum, test_cap, buffer=1.3):
    """Choose hold-out groups (photographers / cameras) for one source so every
    stratum has enough hold-out candidates; all other groups feed training."""
    by_group = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, L in cands_by_stratum.items():
        for c in L:
            by_group[c["group"]][s].append(c)
    groups = sorted(by_group)
    rng.shuffle(groups)
    target = {s: math.ceil(PLAN[s][2] * buffer) for s in cands_by_stratum}
    have = collections.Counter()
    test_groups = set()
    for g in groups:
        need = [s for s in target if have[s] < target[s] and by_group[g].get(s)]
        if not need:
            continue
        test_groups.add(g)
        for s, L in by_group[g].items():
            have[s] += min(len(L), test_cap)
        if all(have[s] >= target[s] for s in target):
            break
    result = {}
    for s in cands_by_stratum:
        n_train, n_test = PLAN[s][1], PLAN[s][2]
        test_pool, train_groups = [], []
        for g in groups:
            L = by_group[g].get(s, [])
            if not L:
                continue
            if g in test_groups:
                test_pool += L[:test_cap]
            else:
                train_groups.append(L)
        # smallest per-group cap that still leaves a 15% download buffer
        cap = 1
        while sum(min(len(L), cap) for L in train_groups) < n_train * 1.15 and cap < max(map(len, train_groups)):
            cap += 1
        train_pool = [c for L in train_groups for c in L[:cap]]
        rng.shuffle(test_pool)
        rng.shuffle(train_pool)
        log(f"split {s}: test pool {len(test_pool)} (need {n_test}) from {sum(1 for g in test_groups if by_group[g].get(s))} groups; "
            f"train pool {len(train_pool)} (need {n_train}) from {len(train_groups)} groups, cap {cap}/group")
        result[s] = {"test": test_pool, "train": train_pool}
    return result


# ---------------------------------------------------------------- download
def ext_for(data):
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return None


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def get_one(c, split, cls):
    d = os.path.join(OUT, split, cls)
    stem = safe(f"{c['stratum'].replace('nacti_', 'nacti-').replace('inat_', 'inat-').replace('nilgai_', 'nilgai-')}_{c['source_id']}")
    for e in ("jpg", "png"):
        if os.path.exists(os.path.join(d, f"{stem}.{e}")):
            return dict(c, split=split, **{"class": cls}, file=f"{split}/{cls}/{stem}.{e}")
    try:
        data = fetch(c["image_url"])
    except Exception as e:  # noqa: BLE001
        log(f"  failed {c['image_url']}: {e}")
        return None
    e = ext_for(data)
    if not e or len(data) < 5000:
        log(f"  rejected {c['image_url']}: not an image or too small ({len(data)} bytes)")
        return None
    path = os.path.join(d, f"{stem}.{e}")
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)
    return dict(c, split=split, **{"class": cls}, file=f"{split}/{cls}/{stem}.{e}")


SEEN = set()   # content hashes already in the set (some photos were uploaded twice)


def take(pool, n, split, cls, ex):
    got, i = [], 0
    while len(got) < n and i < len(pool):
        chunk = pool[i:i + (n - len(got))]
        i += len(chunk)
        for r in ex.map(lambda c: get_one(c, split, cls), chunk):   # results come back in chunk order
            if not r:
                continue
            h = hashlib.sha1(open(os.path.join(OUT, r["file"]), "rb").read()).hexdigest()
            if h in SEEN:
                log(f"  duplicate image skipped: {r['file']}")
                os.remove(os.path.join(OUT, r["file"]))
                continue
            SEEN.add(h)
            got.append(r)
    return got


def main():
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "test"):
        for cls in ("tortoise", "blank", "other_animal"):
            os.makedirs(os.path.join(OUT, split, cls), exist_ok=True)

    tort = inat_candidates(inat_all("gopherus_berlandieri", "Gopherus berlandieri", {}), "inat_tortoise", "Texas tortoise")
    arma = inat_candidates(inat_all("dasypus_mexicanus_studybox", "Dasypus mexicanus", STUDY_BOX),
                           "inat_armadillo", "nine-banded armadillo")
    plans = {}
    plans.update(split_source({"inat_tortoise": tort, "inat_armadillo": arma}, test_cap=8))
    plans.update(split_source(nacti_candidates(), test_cap=6))
    ng = nilgai_candidates()
    for s, L in ng.items():   # no camera/sequence ids in this sample, so a plain random split
        L = sorted(L, key=lambda c: c["source_id"])
        rng.shuffle(L)
        plans[s] = {"test": L[:PLAN[s][2]], "train": L[PLAN[s][2]:]}

    rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for s, (cls, n_train, n_test) in PLAN.items():
            for split, n in (("test", n_test), ("train", n_train)):
                got = take(plans[s][split], n, split, cls, ex)
                log(f"{s} {split}: {len(got)}/{n}")
                if len(got) < n:
                    log(f"  SHORT by {n - len(got)}")
                rows += got

    cols = ["split", "class", "species", "file", "source", "camera_trap", "group", "source_id", "source_page",
            "image_url", "license", "attribution", "location", "latlon", "date", "stratum"]
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["split"], r["class"], r["file"])):
            w.writerow(r)

    # leakage check: no group in both splits
    gs = collections.defaultdict(set)
    for r in rows:
        gs[r["split"]].add(r["group"])
    overlap = gs["train"] & gs["test"]
    tally = collections.Counter((r["split"], r["class"]) for r in rows)
    log("composition:", {f"{k[0]}/{k[1]}": v for k, v in sorted(tally.items())})
    log("matches paper:", all(tally[k] == v for k, v in PAPER.items()), "| groups in both splits:", len(overlap))


if __name__ == "__main__":
    main()
