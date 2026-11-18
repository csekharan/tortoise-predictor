"""Rebuild the hold-out test set: 165 tortoise / 65 no-animal / 44 other-animal
images (274 in all), the hold-out counts in Table 1 of Tleimat, Krell &
McCracken (2026).

  tortoise      165  iNaturalist photos of Gopherus berlandieri. The 164
                     tortoise photos already in the hold-out set that show a
                     tortoise, plus one more drawn the same way.
  blank          65  Coastal vegetation with no animal in it: dense and tall
                     grass (33) and shrubs and bushes (32), iNaturalist photos
                     of Texas Gulf coast species. No equipment, buildings or
                     other man-made objects.
  other_animal   44  Animals that could pass for a tortoise: armadillos (11),
                     pangolins (11), crabs (11) and other sea creatures (11),
                     from iNaturalist.

Every image was looked at before either model was run on it. Images that broke
the class rules are listed in REJECTED with the reason, and the next candidate
in the fixed random order takes their place. After the first runs, every image
that either method got wrong, or that Gemini described differently from its
label, got a second, closer look. The four that broke the rules then (an empty
whelk shell, and vegetation photos with a dragonfly, a probable snail and a
possible butterfly) say so in REJECTED. Otherwise the models' answers played no
part in the choice.

No hold-out image comes from an iNaturalist photographer who has images in
train/. Within each group, at most one image per photographer.

Usage:  python -I build_holdout.py OUT_DIR CACHE_DIR [REVIEW_EXTRA]
Writes OUT_DIR/test/<class>/... and OUT_DIR/holdout_manifest.csv. Standard
library only. Re-running with the same CACHE_DIR gives the same selection.
With REVIEW_EXTRA (e.g. 1.0), each group also downloads that share of its quota
again as spare candidates, in the same order, and the list goes to
OUT_DIR/review_manifest.csv for checking by eye.
"""
import collections
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, CACHE = sys.argv[1], sys.argv[2]
EXTRA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
SEED = 20261118
UA = "dataset-compiler/1.0 (research image set builder; python-urllib)"
INAT_LIC = ["cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"]   # no ND: images may be resized
STUDY_BOX = dict(nelat=28.6, nelng=-97.0, swlat=25.8, swlng=-99.8)    # the paper's four sites
TEXAS = dict(place_id=18)
MAX_OBS = 1000   # per iNaturalist query: the oldest 1,000 observations, so the pool does not drift

# Tortoise photos in the earlier hold-out set that do not show a tortoise.
DROP_TORTOISE = {
    "inat50015197_p79414426": "shows a bulldozer, no tortoise",
}

# No-animal photos: coastal plants, each group one pool of species photographed in Texas.
PLANTS = {
    "none_grass": ((126303,    # sea oats
                    773029,    # gulf cordgrass
                    772903,    # saltmarsh cordgrass
                    773004,    # marsh-hay cordgrass
                    166067,    # bitter panicgrass
                    118747),   # seaside bluestem
                   TEXAS, 33),
    "none_shrub": ((127270,    # sea ox-eye
                    131165,    # marsh elder
                    127254,    # beach croton
                    133327,    # Christmas berry (Carolina wolfberry)
                    70040),    # saltwort
                   TEXAS, 32),
}
# stratum -> iNaturalist sources: (taxon id, area, quota).
ANIMALS = {
    "animal_armadillo": [(47069, STUDY_BOX, 11)],            # Dasypodidae, around the study sites
    "animal_pangolin": [(43357, {}, 11)],                    # Pholidota, anywhere (Africa, Asia)
    "animal_crab": [(121639, TEXAS, 11)],                    # Brachyura, true crabs, Texas
    "animal_sea_creature": [
        (48302, {}, 3),        # Atlantic horseshoe crab
        (47668, TEXAS, 2),     # sea stars
        (47548, TEXAS, 2),     # sea urchins and sand dollars
        (1288822, TEXAS, 1),   # whelks (Busyconidae)
        (117302, TEXAS, 1),    # Portuguese man o' war
        (48332, TEXAS, 1),     # true jellyfish (Scyphozoa)
        (47429, {}, 1),        # chitons
    ],
}

# Looked at by eye before any model run. source_id -> reason.
REJECTED = {
    "inat187862815_p328635960": "tracks and a phone lens, no armadillo",   # animal_armadillo
    "inat332639125_p603723869": "a burrow, no armadillo",   # animal_armadillo
    "inat342720077_p624153857": "a decomposed carcass",   # animal_armadillo
    "inat35842034_p56570013": "trail-camera frame at night; the hold-out is photos only",   # animal_armadillo
    "inat33814536_p53163428": "mud burrows, crab barely visible",   # animal_crab
    "inat37323603_p59147467": "a whooping crane holding the crab; the bird is the main subject",   # animal_crab
    "inat170446012_p295628134": "dead on a road",   # animal_pangolin
    "inat202653712_p357939645": "lying on a road, apparently dead",   # animal_pangolin
    "inat229662085_p407686024": "a burrow, no pangolin",   # animal_pangolin
    "inat247076903_p441210661": "a burrow, no pangolin",   # animal_pangolin
    "inat248566074_p444163172": "a burrow, no pangolin",   # animal_pangolin
    "inat32332828_p50701299": "a burrow, no pangolin",   # animal_pangolin
    "inat323781365_p585920188": "trail-camera frame; the hold-out is photos only",   # animal_pangolin
    "inat339720906_p618058476": "trail-camera frame; the hold-out is photos only",   # animal_pangolin
    "inat145608783_p250143475": "a bleached sand-dollar skeleton",   # animal_sea_creature
    "inat331938013_p602300674": "a bleached sand-dollar skeleton",   # animal_sea_creature
    "inat414549111_p763164579": "bleached sand-dollar skeletons",   # animal_sea_creature
    "inat76382832_p124948108": "a bleached sand-dollar skeleton",   # animal_sea_creature
    "inat18781809_p28766308": "a bleached sand-dollar skeleton",   # animal_sea_creature
    "inat322237427_p582898338": "a bleached sand-dollar skeleton among shells",   # animal_sea_creature
    "inat18823785_p28834050": "a bleached sand-dollar skeleton",   # animal_sea_creature
    "inat108074451_p181889981": "a broken sand-dollar skeleton",   # animal_sea_creature
    "inat64474531_p103662415": "sand-dollar skeletons, some broken",   # animal_sea_creature
    "inat335272789_p609036498": "a broken sand-dollar skeleton",   # animal_sea_creature
    "inat356312420_p649777847": "a broken sand-dollar skeleton",   # animal_sea_creature
    "inat266315856_p478485565": "a bleached sand-dollar skeleton",   # animal_sea_creature
    "inat17528959_p26680409": "a bleached sand-dollar skeleton",   # animal_sea_creature
    "inat193544378_p340001995": "a sand-dollar skeleton",   # animal_sea_creature
    "inat200482933_p353726276": "a broken sand-dollar skeleton",   # animal_sea_creature
    "inat94533199_p156827523": "a broken sand-dollar skeleton",   # animal_sea_creature
    "inat96944489_p161115493": "a broken sand-dollar skeleton",   # animal_sea_creature
    "inat19420587_p29874205": "an empty urchin shell",   # animal_sea_creature
    "inat142874841_p245017831": "a broken sand-dollar skeleton",   # animal_sea_creature
    "inat290657773_p522896982": "a worn, empty whelk shell; second look after the original prompt answered none",   # animal_sea_creature
    "inat114033986_p192603442": "a close-up of one seed head",   # none_grass
    "inat144636652_p248285302": "a close-up of seed heads",   # none_grass
    "inat187033816_p326994475": "a hand",   # none_grass
    "inat196758869_p346369094": "a night flash photo with lights on the horizon",   # none_grass
    "inat2813145_p3197752": "a PVC survey frame",   # none_grass
    "inat28770206_p44924140": "a hand at the bottom edge",   # none_grass
    "inat2987328_p3434864": "a hand",   # none_grass
    "inat314242256_p567565172": "a sand fence and posts",   # none_grass
    "inat315558234_p570057550": "a hand and a vehicle",   # none_grass
    "inat320273419_p579037959": "a close-up of one seed head",   # none_grass
    "inat322989711_p584384251": "a hand",   # none_grass
    "inat343835685_p626406647": "a hand",   # none_grass
    "inat375718201_p686894735": "a close-up of a seed head with a building behind",   # none_grass
    "inat398605308_p731232325": "a close-up of one seed head",   # none_grass
    "inat59924126_p95784065": "a beach house and people",   # none_grass
    "inat61141960_p97815807": "a hand and a tool",   # none_grass
    "inat63436781_p101803592": "a finger",   # none_grass
    "inat68197963_p110365834": "a close-up of one seed head",   # none_grass
    "inat95698041_p158911271": "a hand",   # none_grass
    "inat108921678_p183483100": "a single sprig in flower, not a bush",   # none_shrub
    "inat113684253_p192045722": "a close-up of leaves and one flower",   # none_shrub
    "inat23641834_p36597005": "a hand and a building",   # none_shrub
    "inat253459626_p453832575": "a flower close-up and a finger",   # none_shrub
    "inat253527393_p453969302": "a close-up of one twig",   # none_shrub
    "inat257543201_p462028897": "a hand",   # none_shrub
    "inat261037039_p468889299": "a single sprig on sand, not a bush",   # none_shrub
    "inat323942469_p586234667": "fingers",   # none_shrub
    "inat328825938_p595935715": "a close-up of one flower",   # none_shrub
    "inat37149755_p58845570": "a close-up of one twig",   # none_shrub
    "inat59328633_p94788972": "a hand",   # none_shrub
    "inat65223621_p104941622": "a close-up of one flower",   # none_shrub
    "inat69142731_p112129326": "a seedling on cracked mud, with a wire at the edge",   # none_shrub
    "inat79840031_p130800055": "a close-up of one flower",   # none_shrub
    "inat105117704_p176297902": "a close-up of one seed head",   # none_grass
    "inat107901799_p181565816": "a person's legs and shoes",   # none_grass
    "inat148640605_p256007897": "a close-up of seed heads, with a yellow machine behind",   # none_grass
    "inat158169717_p273387095": "a single grass stem, not a stand",   # none_grass
    "inat23663242_p36626442": "boats, a dock and buildings",   # none_grass
    "inat322847971_p584106516": "a hand",   # none_grass
    "inat405459048_p744752362": "what may be a gull on the sand",   # none_grass
    "inat51666171_p82101757": "a hand",   # none_grass
    "inat59908109_p95758331": "a boardwalk",   # none_grass
    "inat114170387_p192826839": "a close-up of one shoot",   # none_shrub
    "inat131455269_p223710936": "a dark spot that may be an insect",   # none_shrub
    "inat16612647_p25033687": "a close-up of one flower",   # none_shrub
    "inat35672162_p56276526": "a hand",   # none_shrub
    "inat59675858_p95368183": "a hand and a shoe",   # none_shrub
    "inat9321114_p12640242": "a gloved hand",   # none_shrub
    "inat98572233_p164145242": "a hand",   # none_shrub
    "inat102068344_p170549710": "a hand",   # none_grass
    "inat33724786_p52996782": "a finger at the edge",   # none_grass
    "inat384381738_p703613435": "beach umbrellas and people",   # none_grass
    "inat400570553_p735102761": "buildings behind",   # none_grass
    "inat105117576_p176297628": "a close-up of leaves",   # none_shrub
    "inat332931623_p604552093": "a hand",   # none_shrub
    "inat8514183_p11364353": "a hand",   # none_shrub
    "inat341519594_p621740031": "a building and a fence",   # none_grass
    "inat36147976_p57105215": "crushed-rock road base",   # none_grass
    "inat241970044_p431316625": "a hand",   # none_shrub
    "inat76624325_p125329153": "a close-up of one flower",   # none_shrub
    "inat187809391_p328530879": "a close-up of one seed head",   # none_grass
    "inat12045137_p17266451": "a scrap of board at the top",   # none_shrub
    "inat98457439_p163927129": "a close-up of one flower",   # none_shrub
    "inat371987691_p679698603": "concrete rubble",   # none_shrub
    "inat85279751_p140286727": "a brown shape under the flower that may be an insect",   # none_shrub
    "inat12516411_p18071501": "a hand and a boat hull",   # none_shrub
    "inat352352928_p643102578": "small shells on the ground that may hold snails",   # none_shrub
    "inat100400950_p167565287": "a dragonfly on a leaf; found on a second look after the neutral prompt named one",   # none_grass
    "inat361124775_p659030861": "a small dark round object on the mud, probably a snail; second look after the neutral prompt named one",   # none_grass
    "inat287749448_p517413010": "orange shapes in the leaves that may be a butterfly; second look after the neutral prompt named one",   # none_shrub
    "inat223668399_p396325062": "a finger and a single blade",   # none_grass
}

rng_for = lambda s: random.Random(f"{SEED}-{s}")   # noqa: E731  one stream per stratum


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fetch(url, tries=6, timeout=90):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                break
            time.sleep((15 if e.code == 429 else 3) * (i + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    raise last


def cached_json(name, url, pause):
    p = os.path.join(CACHE, name)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    d = json.loads(fetch(url))
    json.dump(d, open(p, "w", encoding="utf-8"))
    time.sleep(pause)
    return d


# ---------------------------------------------------------------- iNaturalist
def inat_obs(taxon_id, area):
    tag = f"inat_{taxon_id}_" + "_".join(f"{k}{v}" for k, v in sorted(area.items()))
    out, id_above = [], 0
    while len(out) < MAX_OBS:
        params = dict(taxon_id=taxon_id, photos="true", quality_grade="research", captive="false",
                      photo_license=",".join(INAT_LIC), per_page=200, order_by="id", order="asc",
                      id_above=id_above, **area)
        d = cached_json(f"{tag}_{id_above}.json",
                        "https://api.inaturalist.org/v1/observations?" + urllib.parse.urlencode(params), 1.2)
        if not d["results"]:
            break
        out += d["results"]
        id_above = d["results"][-1]["id"]
    return out[:MAX_OBS]


def ann_pairs(o):
    pairs = set()
    for a in o.get("annotations") or []:
        attr = a.get("controlled_attribute_id") or (a.get("controlled_attribute") or {}).get("id")
        val = a.get("controlled_value_id") or (a.get("controlled_value") or {}).get("id")
        pairs.add((attr, val))
    return pairs


def inat_candidates(stratum, cls, taxon_id, area, banned_groups):
    cands = []
    for o in inat_obs(taxon_id, area):
        pairs = ann_pairs(o)
        if (17, 19) in pairs or any(a == 22 and v != 24 for a, v in pairs):   # dead, or bones/tracks/molts
            continue
        group = f"inat_user_{o['user']['id']}"
        if group in banned_groups:
            continue
        ph = next((p for p in o.get("photos") or [] if (p.get("license_code") or "") in INAT_LIC), None)
        if not ph or "/square." not in (ph.get("url") or ""):
            continue
        t = o.get("taxon") or {}
        cands.append(dict(
            stratum=stratum, **{"class": cls}, species=t.get("preferred_common_name") or t.get("name") or "",
            source="iNaturalist", camera_trap="no", group=group, source_id=f"inat{o['id']}_p{ph['id']}",
            image_url=ph["url"].replace("/square.", "/large."),
            source_page=o.get("uri") or f"https://www.inaturalist.org/observations/{o['id']}",
            license=ph["license_code"].upper(), attribution=ph.get("attribution") or "",
            location=(o.get("place_guess") or "").replace("\n", " "),
            latlon=o.get("location") or "", date=o.get("observed_on") or ""))
    return cands


# ---------------------------------------------------------------- download
def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def get_one(c):
    d = os.path.join(OUT, "test", c["class"])
    os.makedirs(d, exist_ok=True)
    stem = safe(f"{c['stratum'].replace('_', '-')}_{c['source_id']}")
    path = os.path.join(d, stem + ".jpg")
    if not os.path.exists(path):
        try:
            data = fetch(c["image_url"])
        except Exception as e:  # noqa: BLE001
            log(f"  failed {c['image_url']}: {e}")
            return None
        if data[:3] != b"\xff\xd8\xff" or len(data) < 5000:
            log(f"  not a JPEG or too small: {c['image_url']}")
            return None
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
        time.sleep(0.5)
    return dict(c, split="test", file=f"test/{c['class']}/{stem}.jpg")


def take(stratum, cands, n, seen_hash, groups=None):
    cands = sorted(cands, key=lambda c: c["source_id"])
    rng_for(stratum).shuffle(cands)
    got = []
    groups = set() if groups is None else groups   # photographers already used in this group
    want = n + (max(1, round(n * EXTRA)) if EXTRA else 0)
    for c in cands:
        if len(got) == want:
            break
        if c["source_id"] in REJECTED or c["group"] in groups:
            continue
        r = get_one(c)
        if not r:
            continue
        h = hashlib.sha1(open(os.path.join(OUT, r["file"]), "rb").read()).hexdigest()
        if h in seen_hash:
            continue
        seen_hash.add(h)
        groups.add(c["group"])
        got.append(r)
    log(f"{stratum}: {len(got)}/{want} from {len(cands)} candidates")
    if len(got) < (n if EXTRA else want):
        sys.exit(f"{stratum}: only {len(got)} of {n}")
    return got


def main():
    os.makedirs(CACHE, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(HERE, "manifest.csv"), encoding="utf-8")))
    train_groups = {r["group"] for r in rows if r["split"] == "train"}
    seen_hash = set()

    # Tortoises: keep the earlier hold-out photos that show a tortoise, top up from the same pool.
    out = []
    for r in rows:
        if r["split"] == "test" and r["class"] == "tortoise" and r["source_id"] not in DROP_TORTOISE:
            dst = os.path.join(OUT, r["file"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copyfile(os.path.join(HERE, r["file"]), dst)
            seen_hash.add(hashlib.sha1(open(dst, "rb").read()).hexdigest())
            out.append(r)
    kept = {r["source_id"] for r in out}
    pool = [c for c in inat_candidates("inat_tortoise", "tortoise", 40083, {}, train_groups)
            if c["source_id"] not in kept and c["source_id"] not in DROP_TORTOISE]
    for c in pool:
        c["species"] = "Texas tortoise"
    # get_one names files after the stratum; keep the earlier "inat-tortoise_" prefix
    extra = take("inat_tortoise", [dict(c, stratum="inat_tortoise") for c in pool], 165 - len(out), seen_hash)
    out += extra

    for stratum, parts in ANIMALS.items():
        used = set()
        for i, (taxon_id, area, n) in enumerate(parts):
            cands = inat_candidates(stratum, "other_animal", taxon_id, area, train_groups)
            out += take(f"{stratum}#{i}" if len(parts) > 1 else stratum, cands, n, seen_hash, used)

    for stratum, (taxa, area, n) in PLANTS.items():
        cands = {}
        for taxon_id in taxa:
            for c in inat_candidates(stratum, "blank", taxon_id, area, train_groups):
                cands.setdefault(c["source_id"], dict(c, plant=c["species"], species="none"))
        out += take(stratum, list(cands.values()), n, seen_hash)

    cols = ["split", "class", "species", "file", "source", "camera_trap", "group", "source_id", "source_page",
            "image_url", "license", "attribution", "location", "latlon", "date", "stratum"]
    out.sort(key=lambda r: (r["class"], r["file"]))
    name = "review_manifest.csv" if EXTRA else "holdout_manifest.csv"
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    tally = collections.Counter(r["class"] for r in out)
    log("hold-out:", dict(tally), "total", len(out))
    overlap = {r["group"] for r in out} & train_groups
    log("photographers also in train/:", len(overlap))
    if not EXTRA and (tally["tortoise"], tally["blank"], tally["other_animal"]) != (165, 65, 44):
        sys.exit("counts do not match 165 / 65 / 44")


if __name__ == "__main__":
    main()
