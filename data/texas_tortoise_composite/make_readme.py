"""Write README.md for the texas_tortoise_composite folder from its manifest.csv.
Usage: python -I make_readme.py OUT_DIR"""
import collections
import csv
import os
import sys

OUT = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(OUT, "manifest.csv"), encoding="utf-8")))
C = collections.Counter
PAPER = {("train", "tortoise"): 1332, ("train", "blank"): 420, ("train", "other_animal"): 953,
         ("test", "tortoise"): 165, ("test", "blank"): 65, ("test", "other_animal"): 44}
CLS = ["tortoise", "blank", "other_animal"]
SHORT = {"iNaturalist": "iNaturalist photos (hand-held)",
                  "NACTI (LILA BC), Archbold FL": "NACTI trail cameras, Archbold, Florida",
         "Kutugata et al. 2021 sample (github.com/mkutu/Nilgai), South Texas": "Nilgai-study trail cameras, South Texas"}
GROUPS = [  # hold-out stratum, label, what it holds
    ("inat_tortoise", "Texas tortoise", "*Gopherus berlandieri*, wild, Texas and northern Mexico"),
    ("none_grass", "No animal: dense and tall grass",
     "sea oats, gulf, saltmarsh and marsh-hay cordgrass, bitter panicgrass, seaside bluestem"),
    ("none_shrub", "No animal: shrubs and bushes", "sea ox-eye, marsh elder, beach croton, Christmas berry, saltwort"),
    ("animal_armadillo", "Armadillos", "nine-banded armadillos photographed around the paper's four study sites"),
    ("animal_pangolin", "Pangolins", "Africa and Asia"),
    ("animal_crab", "Crabs", "true crabs photographed in Texas"),
    ("animal_sea_creature", "Other sea creatures", ""),
]

sc = C((r["split"], r["class"]) for r in rows)
src = open(os.path.join(OUT, "build_holdout.py"), encoding="utf-8").read()
i = src.index("REJECTED = {")
N_REJ = C("blank" if "# none_" in line else "animal"
          for line in src[i:src.index("\n}\n", i)].splitlines() if line.strip().startswith('"'))
test =[r for r in rows if r["split"] == "test"]
train = [r for r in rows if r["split"] == "train"]
L = []
w = L.append
w("# Texas tortoise composite image set\n")
w("This folder holds public images arranged to match the training and hold-out image counts that "
  "Tleimat, Krell and McCracken (2026) report for their Texas tortoise detector (Table 1 of "
  "\"Needle in a haystack: towards the use of machine learning to detect Texas tortoises in large "
  "datasets\", *Amphibian & Reptile Conservation* 20(1): 67-81). The paper's own images were never "
  "published, so these are different images chosen to fill the same slots: the same number of "
  "tortoise, no-animal and other-animal pictures in the training set (the paper's model M4, the one "
  "released on GitHub) and in the hold-out test set.\n")
w("## Composition\n")
w("| Split | Tortoise | No animal (`blank/`) | Other animal | Total | Paper (Table 1) |")
w("|---|---:|---:|---:|---:|---|")
for sp, lab in (("train", "Training"), ("test", "Hold-out test")):
    n = [sc[(sp, c)] for c in CLS]
    p = [PAPER[(sp, c)] for c in CLS]
    w(f"| {lab} | {n[0]:,} | {n[1]:,} | {n[2]:,} | {sum(n):,} | {p[0]:,} / {p[1]:,} / {p[2]:,} = {sum(p):,} |")
tot = [sc[("train", c)] + sc[("test", c)] for c in CLS]
w(f"| All | {tot[0]:,} | {tot[1]:,} | {tot[2]:,} | {sum(tot):,} | |\n")

w("## Hold-out test set\n")
w("Every hold-out image is a photo, not a trail-camera frame, and every one was looked at before any "
  "model was run on it.\n")
w("| Class | Group | Images | Source | What it shows |")
w("|---|---|---:|---|---|")
for key, label, note in GROUPS:
    rr = [r for r in test if r["stratum"] == key]
    if key == "animal_sea_creature":
        note = ", ".join(f"{s} ({n})" for s, n in C(r["species"] for r in rr).most_common())
    cls = rr[0]["class"] if rr else ""
    srcs = " and ".join(sorted({SHORT.get(r["source"], r["source"]) for r in rr}))
    w(f"| `{cls}` | {label} | {len(rr)} | {srcs} | {note} |")
w("")
for key, label, _ in GROUPS:
    if key in ("animal_crab", "animal_pangolin"):
        names = C(r["species"] for r in test if r["stratum"] == key).most_common()
        w(f"- **{label}:** " + ", ".join(f"{s} ({n})" for s, n in names) + ".")
w("")
w("iNaturalist now files the Texas nine-banded armadillo as *Dasypus mexicanus*, \"Mexican long-nosed "
  "armadillo\", and that is the name in the `species` column.\n")
w("**How the hold-out images were chosen.** `build_holdout.py OUT_DIR CACHE_DIR` (Python, standard "
  "library only) rebuilds the hold-out set:\n")
w("- **Tortoises.** The tortoise photos are research-grade iNaturalist observations of *Gopherus "
  "berlandieri*, chosen as described under the training set below. An earlier version of this hold-out "
  "set had one photo labelled tortoise that shows a bulldozer and no tortoise; it was replaced with "
  "another tortoise photo drawn the same way.")
w("- **No animal: dense and tall grass, shrubs and bushes.** Research-grade iNaturalist observations "
  "of coastal plants photographed in Texas: grasses of the dunes and salt marshes (sea oats, the "
  "cordgrasses, bitter panicgrass, seaside bluestem) and shrubs of the shore and marsh edge (sea ox-eye, "
  "marsh elder, beach croton, Christmas berry, saltwort). Each group's species are pooled and drawn "
  "together, chosen like the animal photos below. These are hand-held photos, taken the way the "
  "tortoise photos were.")
w("- **Other animals.** Research-grade iNaturalist observations, wild, with a CC0, CC BY, CC BY-NC, "
  "CC BY-SA or CC BY-NC-SA photo, not annotated as dead or as bones, tracks or moults. Armadillos come "
  "from the box around the paper's four study sites, crabs and most sea creatures from Texas, and "
  "pangolins, horseshoe crabs and chitons from anywhere.")
w("- **Order and checks.** Each group's candidates are shuffled with a fixed seed and taken in that "
  "order, at most one image per photographer in each group. No hold-out image comes "
  "from an iNaturalist photographer who has images in `train/`. Each picked image was then looked at, "
  "and the next candidate took the place of any that broke these rules:")
w("  - A no-animal photo must show the vegetation itself, a stand of grass or a shrub or bush, not a "
  "close-up of one flower, seed head or leaf. It must show no animal at all (no insects, spiders, "
  "snails, birds, people, hands or feet) and no equipment or other man-made object: no vehicles, "
  "machinery, tools, rulers, signs, fences, posts, boardwalks, buildings, roads or litter.")
w("  - An animal photo must show the animal itself, not a burrow, tracks, a carcass, a bleached "
  "skeleton (most Texas sand-dollar photos are skeletons) or a trail-camera frame, and no other animal "
  "may be the main subject.\n")
w(f"`REJECTED` in the script lists all {sum(N_REJ.values())} rejected candidates, {N_REJ['blank']} "
  f"no-animal photos and {N_REJ['animal']} animal photos, each with its reason.\n")
w("The models played no part in choosing images, with one exception. After the first runs, every image "
  "that either method got wrong, or that Gemini described differently from its label, got a second, "
  "closer look. Four broke the rules and were replaced by the next candidates: a worn, empty whelk "
  "shell, and three vegetation photos in which Gemini's neutral prompt named an animal. On a closer "
  "look one had a dragonfly on a leaf, one a small dark round object on the mud that is probably a "
  "snail, and one orange shapes in the leaves that may be a butterfly. Their entries in `REJECTED` say "
  "so. The vegetation photos that Mask R-CNN called a tortoise got the same second look; none hid an "
  "animal.\n")
w("An earlier version of the no-animal class used dry-land and beach photos from Wikimedia Commons. It "
  "was replaced with coastal vegetation only, with no equipment or other man-made objects in frame.\n")

w("## Training set\n")
w("The training set is the one built for the paper's model M4 and is not used by the comparison in "
  "this repository; it is there for retraining or few-shot prompting. `build_tortoise_dataset.py "
  "OUT_DIR CACHE_DIR` downloads it, and `crop_camera_strips.ps1` then trims the camera strips "
  "described below. (That script also built the first version of the hold-out set, which "
  "`build_holdout.py` replaced.)\n")
w("| Class | Source | Images |")
w("|---|---|---:|")
for (cl, src), n in sorted(C((r["class"], r["source"]) for r in train).items(),
                           key=lambda kv: (CLS.index(kv[0][0]), kv[0][1])):
    w(f"| {cl} | {SHORT.get(src, src)} | {n:,} |")
w("")
w("The training set's other-animal images, by species:\n")
w("| Species | Images | Hand-held photos | Trail-camera frames |")
w("|---|---:|---:|---:|")
oa = [r for r in train if r["class"] == "other_animal"]
for spc in sorted({r["species"] for r in oa}, key=lambda s: -sum(1 for r in oa if r["species"] == s)):
    rr = [r for r in oa if r["species"] == spc]
    w(f"| {spc} | {len(rr)} | {sum(r['camera_trap'] == 'no' for r in rr)} | "
      f"{sum(r['camera_trap'] == 'yes' for r in rr)} |")
w("")
w("**Tortoises.** Research-grade iNaturalist observations of *Gopherus berlandieri* (wild, from "
  "Texas and northern Mexico) whose photo carries a CC0, CC BY, CC BY-NC, CC BY-SA or CC BY-NC-SA "
  "licence. No-derivatives licences were left out so the images can be resized and augmented. "
  "Observations annotated as dead, or as bones or tracks rather than the animal, were dropped. One "
  "photo per observation, at iNaturalist's \"large\" size (1,024 pixels on the long side). The paper "
  "also drew some of its tortoise images from iNaturalist.\n")
w("**Blank frames.** Trail-camera frames labelled empty by their source: from the South Texas sample "
  "that Kutugata et al. published with their nilgai model (the \"South Texas model\" the paper "
  "compares against), and from NACTI's cameras at Archbold, Florida. At most one frame per trigger "
  "burst. These labels were not checked by eye, and a few frames do show an animal.\n")
w("**Other animals.** As the paper did, armadillos get extra weight because they are the animal "
  "most likely to be mistaken for a tortoise. Half the armadillos are iNaturalist photos taken "
  "inside a box around the paper's four study sites; the other half are NACTI trail-camera frames. "
  "The rest are trail-camera frames of animals that also live in South Texas. At most one frame per "
  "trigger burst and three per camera per day.\n")
w("**Camera strips trimmed.** Every NACTI frame came from a Reconyx camera that stamps a black "
  "strip with date and temperature along the top and the camera number and logo along the bottom. "
  "The tortoise photos have no such strips, so a model could have learned \"black strip means no "
  "tortoise\". The top 36 and bottom 64 pixels of each NACTI frame were cut off, leaving "
  "2048 x 1436 pixels, and the frame was re-saved as JPEG at quality 95. The South Texas frames "
  "arrived already without strips (669 x 346 pixels). The `image_url` column still points to the "
  "untrimmed originals.\n")

w("## Folder layout\n")
w("`train/` and `test/` each contain `tortoise/`, `blank/` (no animal) and `other_animal/`. That is "
  "the one-folder-per-class layout that PyTorch `ImageFolder` and Keras `image_dataset_from_directory` "
  "read directly. `manifest.csv` has one row per image: split, class, species, file path, source, "
  "whether it is a trail-camera frame, the photographer or camera it came from "
  "(`group`), the original URL and page, licence, attribution, place, date and the group within its "
  "class (`stratum`).\n")

w("## How this differs from the paper's images\n")
w("The paper's tortoise images were mainly frames from its own trail cameras: Browning Dark "
  "Ops Pro DCL cameras about 40 cm off the ground, tilted about 45 degrees down, taking a picture "
  "every two minutes in daylight, at four South Texas sites in 2022 and 2023. It topped these up with "
  "iNaturalist photos and Reconyx photos from highway wildlife crossings, but gives no count per "
  "source. None of its trail-camera frames are public, and iNaturalist has almost no camera-trap "
  "pictures of this species, so every tortoise here is a hand-held photo, often a close-up. In the "
  "hold-out set every other image is a photo too, so a model cannot tell the classes apart by camera "
  "type there. In the training set the blanks and most other animals are trail-camera frames, so a "
  "model trained on it could learn \"hand-held photo means tortoise\". Scores on the hold-out set "
  "will be higher than the same model would earn on real trail-camera footage. To measure that, you "
  "need the authors' frames.\n")
w("The paper's negatives were trail-camera frames from its own sites. The hold-out negatives here "
  "are chosen to be hard in a different way: animals with a hard shell or a domed back, and stands of "
  "the coastal grasses and shrubs a tortoise could hide in. The paper also gave the habitat "
  "mix of its images (Table 5: Tamaulipan thornscrub 678, brush 210, grassland/prairie 465, bare "
  "ground 137); that mix is not reproduced. The paper's training images carried hand-drawn tortoise "
  "outlines (CVAT polygons, COCO format). This set has image-level labels only, with no boxes or "
  "outlines.\n")

w("## Licences and credit\n")
w("| Licence | Images |")
w("|---|---:|")
for lic, n in C(r["license"] for r in rows).most_common():
    w(f"| {lic} | {n:,} |")
w("")
w("iNaturalist photos keep their photographers' Creative Commons licences. The `attribution` column "
  "holds each credit line. Many iNaturalist "
  "photos are non-commercial (NC), so this set is for research use. NACTI is released under the "
  "Community Data License Agreement (permissive); cite Tabak MA et al. 2019, \"Machine learning to "
  "classify animal species in camera trap images: Applications in ecology\", *Methods in Ecology and "
  "Evolution* 10(4): 585-590. The South Texas sample is MIT licensed, copyright 2020 Matthew Kutugata "
  "(github.com/mkutu/Nilgai); see Kutugata et al. 2021, *Journal of Fish and Wildlife Management* "
  "12(2): 412-421.\n")

open(os.path.join(OUT, "README.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L))
print("\n".join(L[:30]))
