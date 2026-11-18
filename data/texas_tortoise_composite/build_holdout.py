"""Rebuild the hold-out test set: 165 tortoise / 65 no-animal / 44 other-animal
images (274 in all), the hold-out counts in Table 1 of Tleimat, Krell &
McCracken (2026).

  tortoise      165  iNaturalist photos of Gopherus berlandieri. The 164
                     tortoise photos already in the hold-out set that show a
                     tortoise, plus one more drawn the same way.
  blank          65  Photos with no animal in them: vegetation and bushes (22,
                     iNaturalist plant photos from around the paper's study
                     sites), dry land (21) and beaches (22), from Wikimedia
                     Commons.
  other_animal   44  Animals that could pass for a tortoise: armadillos (11),
                     pangolins (11), crabs (11) and other sea creatures (11),
                     from iNaturalist.

Every image was looked at before either model was run on it. Images that broke
the class rules are listed in REJECTED with the reason, and the next candidate
in the fixed random order takes their place. After the first runs, every image
that either method got wrong, or that Gemini described differently from its
label, got a second, closer look; the three that broke the rules then are marked
as such in REJECTED. Otherwise the models' answers played no part in the choice.

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
import html
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

# stratum -> iNaturalist sources: (taxon id, area, quota).
PLANTS = {
    "none_vegetation": [(47126, STUDY_BOX, 22)],             # Plantae, around the study sites
}
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
SCENES = {   # Wikimedia Commons full-text searches, photos only
    "none_dry_land": (21, [
        "cracked mud", "dried mud cracks", "caliche", "bare soil", "desert pavement", "gravel road Texas",
        "dry creek bed Texas", "Big Bend desert", "sandy soil", "dirt road Texas", "rocky ground", "dry riverbed",
        "dry lake bed", "mud cracks", "hardpan", "arid landscape", "dry wash", "gravel bar", "eroded soil",
        "Chihuahuan Desert landscape"]),
    "none_beach": (22, [
        "Padre Island beach", "Mustang Island beach", "Port Aransas beach", "Galveston beach",
        "South Padre Island beach", "Boca Chica beach Texas", "Texas beach dunes", "sargassum beach",
        "seaweed on beach", "driftwood on beach", "beach wrack", "sand dunes beach",
        "Padre Island National Seashore", "Matagorda beach", "Bolivar Peninsula beach", "Surfside Beach Texas",
        "sea oats dunes", "beach sand", "wrack line", "beach at low tide"]),
}
COMMONS_LIC = re.compile(r"^(CC0|Public domain|PD|CC BY(-SA)? \d\.\d)", re.I)
# Titles that name an animal, a person, a building, a vehicle or a non-photo are skipped unseen.
COMMONS_SKIP = re.compile(
    r"map|sign|logo|diagram|chart|illustration|aerial|satellite|night|panorama|museum|specimen|herbarium|"
    r"drawing|painting|poster|\bbees?\b|bird|snake|crotalus|deer|cattle|\bcows?\b|horse|\bdogs?\b|\bcats?\b|"
    r"turtle|tortoise|lizard|butterfl|\bmoth|insect|spider|fish|crab|heron|egret|pelican|gull|\btern|plover|"
    r"sandpiper|hawk|\bowl|coyote|javelina|\bhog|armadillo|people|\bman\b|woman|child|soldier|volunteer|"
    r"visitor|tourist|shop|store|downtown|church|house|cabin|building|school|street|city|hall|courthouse|hotel|"
    r"restaurant|\bcar\b|truck|vehicle|boat|ship|pier|festival|parade|station|bridge|tower|lighthouse|"
    r"patent|newspaper|document|cemetery|parking|silo|goat|marker|\bcamp|\btents?\b|\d{6}-[A-Z]-", re.I)

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
    "commons10083186": "old stereo card of a building",   # none_beach
    "commons105608058": "buses and people",   # none_beach
    "commons107940275": "people",   # none_beach
    "commons109878532": "a person sitting on the beach",   # none_beach
    "commons113277268": "trucks and earth movers",   # none_beach
    "commons113577591": "buildings and a road",   # none_beach
    "commons120105942": "no beach in view",   # none_beach
    "commons141198694": "a sign at the edge",   # none_beach
    "commons148539906": "a pier with amusement rides",   # none_beach
    "commons150024135": "buildings and people",   # none_beach
    "commons161131398": "unclear object, may be animal remains",   # none_beach
    "commons17115699": "people under umbrellas",   # none_beach
    "commons182919662": "people in the water",   # none_beach
    "commons199633463": "a leaf on a black backdrop, not a beach",   # none_beach
    "commons24308366": "pelicans",   # none_beach
    "commons27511303": "a person lying on a log",   # none_beach
    "commons34079627": "a warship",   # none_beach
    "commons37053372": "underwater, not a beach",   # none_beach
    "commons480077": "people on the jetty",   # none_beach
    "commons52571593": "buildings and cars",   # none_beach
    "commons54585425": "shorebirds on the sand",   # none_beach
    "commons61974544": "a boat wreck",   # none_beach
    "commons63770888": "a boat and litter",   # none_beach
    "commons7580806": "a person in the water",   # none_beach
    "commons83883272": "a plant close-up, not a beach",   # none_beach
    "commons110958185": "people",   # none_dry_land
    "commons111097074": "unclear, looks like an aerial view",   # none_dry_land
    "commons132083319": "a micrograph with a scale bar",   # none_dry_land
    "commons142818156": "a sign and a stall",   # none_dry_land
    "commons142883044": "people",   # none_dry_land
    "commons144921904": "goats",   # none_dry_land
    "commons157974616": "snow beside a road, not dry land",   # none_dry_land
    "commons17221927": "people",   # none_dry_land
    "commons17810883": "grain silos",   # none_dry_land
    "commons184067229": "a person",   # none_dry_land
    "commons188379659": "a patent document",   # none_dry_land
    "commons190706782": "a patent document",   # none_dry_land
    "commons191440793": "a patent document",   # none_dry_land
    "commons195213245": "a garden and a building",   # none_dry_land
    "commons36350198": "a road sign",   # none_dry_land
    "commons49985302": "a flower close-up, not dry land",   # none_dry_land
    "commons54704443": "a person",   # none_dry_land
    "commons56227836": "a historical marker",   # none_dry_land
    "commons72464582": "a building and a hammer",   # none_dry_land
    "commons82029978": "painted marks and asphalt",   # none_dry_land
    "commons8632484": "vehicles",   # none_dry_land
    "inat76382832_p124948108": "a bleached sand-dollar skeleton",   # animal_sea_creature
    "commons106402134": "a house and boats",   # none_beach
    "commons120105943": "a park entrance with signs",   # none_beach
    "commons129559351": "a flower bud close-up, not a beach",   # none_beach
    "commons133441953": "a handwritten letter",   # none_beach
    "commons175010482": "people or vehicles on the far beach",   # none_beach
    "commons18505839": "buildings",   # none_beach
    "commons24308354": "gulls",   # none_beach
    "commons24476867": "a bookshop",   # none_beach
    "commons25481754": "a satellite view",   # none_beach
    "commons27330468": "buildings",   # none_beach
    "commons57282295": "a motel and cars",   # none_beach
    "commons61974546": "flooded houses from the air",   # none_beach
    "commons7606246": "a tarantula",   # none_beach
    "commons79388945": "a dark shape on the sand that may be a seal",   # none_beach
    "commons80046097": "gulls and pelicans",   # none_beach
    "commons88788363": "people walking on the beach",   # none_beach
    "commons92551076": "unclear orange objects that may be animals",   # none_beach
    "commons109553146": "a stone bridge",   # none_dry_land
    "commons110958221": "people",   # none_dry_land
    "commons113835485": "green farmland, not dry land",   # none_dry_land
    "commons129835603": "a highway and buildings in the valley",   # none_dry_land
    "commons139243506": "a falcon",   # none_dry_land
    "commons159123385": "a paint box and sketchbook",   # none_dry_land
    "commons188379665": "a patent document",   # none_dry_land
    "commons190706769": "a patent drawing",   # none_dry_land
    "commons46228246": "a lake, not dry land",   # none_dry_land
    "commons46982970": "people",   # none_dry_land
    "commons49985319": "a flower close-up, not dry land",   # none_dry_land
    "commons70447801": "a person's legs",   # none_dry_land
    "commons86372507": "a river, not dry land",   # none_dry_land
    "inat1140944_p1435328": "an insect on the flowers",   # none_vegetation
    "inat1287507_p1614883": "a finger at the edge",   # none_vegetation
    "inat1858351_p2260380": "a hand",   # none_vegetation
    "inat2283873_p2556973": "an arm and hand",   # none_vegetation
    "inat2284152_p2557274": "fingers",   # none_vegetation
    "inat2362611_p2628695": "fingers",   # none_vegetation
    "inat18781809_p28766308": "a bleached sand-dollar skeleton",   # animal_sea_creature
    "commons12073651": "sandpipers",   # none_beach
    "commons18948453": "a stone tower",   # none_beach
    "commons24308359": "a person fishing",   # none_beach
    "commons52571989": "gulls and people",   # none_beach
    "commons106919829": "buildings",   # none_dry_land
    "commons131041791": "a white animal at the right edge",   # none_dry_land
    "commons137027107": "a polished stone slice, not a scene",   # none_dry_land
    "commons6448004": "astronauts and a jet",   # none_dry_land
    "inat1556729_p1927240": "a hand",   # none_vegetation
    "inat2284012_p2557136": "a hand",   # none_vegetation
    "inat2284195_p2557335": "a gloved hand",   # none_vegetation
    "inat2358348_p2623108": "a hand",   # none_vegetation
    "inat322237427_p582898338": "a bleached sand-dollar skeleton among shells",   # animal_sea_creature
    "commons18551878": "a boardwalk, a lifeguard stand and people",   # none_beach
    "commons65175343": "people and a canoe",   # none_beach
    "commons137027115": "a polished stone slice, not a scene",   # none_dry_land
    "inat1619563_p1988684": "a hand",   # none_vegetation
    "inat2350394_p2611210": "a seed on a white backdrop, not a scene",   # none_vegetation
    "inat18823785_p28834050": "a bleached sand-dollar skeleton",   # animal_sea_creature
    "commons17515163": "poles and buildings",   # none_beach
    "commons61974363": "a helicopter",   # none_beach
    "inat2314211_p2574452": "hands",   # none_vegetation
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
    "commons121719688": "a man among storm debris",   # none_beach
    "commons107940276": "a person and a dog far down the beach",   # none_beach
    "commons132505340": "small snails on the seaweed",   # none_beach
    "commons148281377": "a bird in the sky",   # none_beach
    "commons175010283": "vehicles on the far beach",   # none_beach
    "commons73827245": "a small purple creature may be in the seaweed",   # none_beach
    "commons176052281": "weaver-bird nests in the tree",   # none_dry_land
    "commons68792749": "a dark shape at the left that may be cattle",   # none_dry_land
    "commons3835525": "people fishing and buildings",   # none_beach
    "commons49106420": "kangaroos",   # none_dry_land
    "commons84623913": "a fossil coral, animal remains",   # none_dry_land
    "commons17534115": "people on the rocks",   # none_beach
    "commons83153253": "a shoe at the bottom edge",   # none_dry_land
    "commons110061637": "a small white shape at the top that may be a bird",   # none_beach
    "commons51154533": "people on a motorbike",   # none_dry_land
    "commons8074046": "trucks, power lines and a worker",   # none_beach
    "commons26306590": "a sea-turtle hatchling on the reeds; found on a second look after both Gemini prompts named it",   # none_beach
    "commons37392504": "a small dark shape on the sand that may be a grasshopper; second look after the neutral prompt named one",   # none_beach
    "inat290657773_p522896982": "a worn, empty whelk shell; second look after the original prompt answered none",   # animal_sea_creature
    "commons158413806": "a motel and cars",   # none_beach
    "commons24308349": "a person fishing",   # none_beach
    "commons199633447": "a shrub, not a beach",   # none_beach
    "commons8073520": "a man among storm debris",   # none_beach
    "commons23006417": "people",   # none_beach
    "commons1333686": "whelk egg cases, shells and feathers in the wrack",   # none_beach
    "commons49886824": "white specks on the mudflat that may be gulls, and a figure in the water",   # none_beach
    "commons201359016": "houses behind the dune",   # none_beach
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


# ---------------------------------------------------------------- Wikimedia Commons
def strip_html(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def commons_candidates(stratum, queries, claimed):
    """claimed: page ids an earlier group's searches already returned; each page belongs to one group."""
    by_id = {}
    for q in queries:
        params = dict(action="query", format="json", generator="search", gsrsearch=f"{q} filetype:bitmap",
                      gsrnamespace=6, gsrlimit=50, prop="imageinfo", iiprop="url|size|mime|extmetadata",
                      iiurlwidth=960, maxlag=5)
        d = cached_json("commons_" + re.sub(r"\W+", "_", q) + ".json",
                        "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params), 3)
        for page in (d.get("query") or {}).get("pages", {}).values():
            by_id.setdefault(page["pageid"], page)
    cands = []
    for pid, page in sorted(by_id.items()):
        if pid in claimed:
            continue
        claimed.add(pid)
        ii = (page.get("imageinfo") or [{}])[0]
        md = ii.get("extmetadata") or {}
        lic = strip_html((md.get("LicenseShortName") or {}).get("value"))
        title = page["title"]
        if ii.get("mime") != "image/jpeg" or ii.get("width", 0) < 1000 or ii.get("height", 0) < 650:
            continue
        if not 0.5 <= ii["width"] / ii["height"] <= 2.0 or not COMMONS_LIC.match(lic) or COMMONS_SKIP.search(title):
            continue
        artist = strip_html((md.get("Artist") or {}).get("value")) or "unknown"
        lat, lon = (md.get("GPSLatitude") or {}).get("value"), (md.get("GPSLongitude") or {}).get("value")
        cands.append(dict(
            stratum=stratum, **{"class": "blank"}, species="none", source="Wikimedia Commons", camera_trap="no",
            group="commons_" + re.sub(r"\W+", "_", artist.lower()).strip("_")[:60], source_id=f"commons{pid}",
            image_url=ii.get("thumburl") or ii["url"], source_page=ii.get("descriptionurl") or "",
            license=re.sub(r"\s+", "-", lic).upper().replace("PUBLIC-DOMAIN", "Public domain"),
            attribution=f"{artist}, {lic}, via Wikimedia Commons",
            location="", latlon=f"{lat},{lon}" if lat and lon else "",
            date=strip_html((md.get("DateTimeOriginal") or {}).get("value"))[:10]))
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

    for stratum, parts in list(ANIMALS.items()) + list(PLANTS.items()):
        cls = "blank" if stratum in PLANTS else "other_animal"
        used = set()
        for i, (taxon_id, area, n) in enumerate(parts):
            cands = inat_candidates(stratum, cls, taxon_id, area, train_groups)
            if cls == "blank":
                for c in cands:
                    c["species"] = "none"
            out += take(f"{stratum}#{i}" if len(parts) > 1 else stratum, cands, n, seen_hash, used)

    claimed = set()
    for stratum, (n, queries) in SCENES.items():
        out += take(stratum, commons_candidates(stratum, queries, claimed), n, seen_hash)

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
