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

sc = C((r["split"], r["class"]) for r in rows)
L = []
w = L.append
w("# Texas tortoise composite image set\n")
w("This folder holds public images arranged to match the training and hold-out image counts that "
  "Tleimat, Krell and McCracken (2026) report for their Texas tortoise detector (Table 1 of "
  "\"Needle in a haystack: towards the use of machine learning to detect Texas tortoises in large "
  "datasets\", *Amphibian & Reptile Conservation* 20(1): 67-81). The paper's own images were never "
  "published, so these are different images chosen to fill the same slots: the same number of "
  "tortoise, blank and other-animal pictures in the training set (the paper's model M4, the one "
  "released on GitHub) and in the hold-out test set.\n")
w("## Composition\n")
w("| Split | Tortoise | Blank (no animal) | Other animal | Total | Paper (Table 1) |")
w("|---|---:|---:|---:|---:|---|")
for sp, lab in (("train", "Training"), ("test", "Hold-out test")):
    n = [sc[(sp, c)] for c in CLS]
    p = [PAPER[(sp, c)] for c in CLS]
    w(f"| {lab} | {n[0]:,} | {n[1]:,} | {n[2]:,} | {sum(n):,} | {p[0]:,} / {p[1]:,} / {p[2]:,} = {sum(p):,} |")
tot = [sc[("train", c)] + sc[("test", c)] for c in CLS]
w(f"| All | {tot[0]:,} | {tot[1]:,} | {tot[2]:,} | {sum(tot):,} | |\n")

w("Where each slot's images come from:\n")
w("| Split | Class | Source | Images |")
w("|---|---|---|---:|")
for (sp, cl, src), n in sorted(C((r["split"], r["class"], r["source"]) for r in rows).items(),
                                key=lambda kv: (kv[0][0] != "train", CLS.index(kv[0][1]), kv[0][2])):
    w(f"| {sp} | {cl} | {SHORT.get(src, src)} | {n:,} |")
w("")

w("The other-animal images, by species:\n")
w("| Species | Training | Hold-out | Hand-held photos | Trail-camera frames |")
w("|---|---:|---:|---:|---:|")
oa = [r for r in rows if r["class"] == "other_animal"]
for spc in sorted({r["species"] for r in oa}, key=lambda s: -sum(1 for r in oa if r["species"] == s)):
    rr = [r for r in oa if r["species"] == spc]
    w(f"| {spc} | {sum(r['split'] == 'train' for r in rr)} | {sum(r['split'] == 'test' for r in rr)} | "
      f"{sum(r['camera_trap'] == 'no' for r in rr)} | {sum(r['camera_trap'] == 'yes' for r in rr)} |")
w("")

w("## Folder layout\n")
w("`train/` and `test/` each contain `tortoise/`, `blank/` and `other_animal/`. That is the "
  "one-folder-per-class layout that PyTorch `ImageFolder` and Keras `image_dataset_from_directory` "
  "read directly. `manifest.csv` has one row per image: split, class, species, file path, source, "
  "whether it is a trail-camera frame, the photographer or camera it came from (`group`), the "
  "original URL and page, licence, attribution, place and date. Two scripts rebuild the folder: "
  "`build_tortoise_dataset.py OUT_DIR CACHE_DIR` (Python, standard library only) downloads the same "
  "selection, and `crop_camera_strips.ps1` then trims the camera strips described below.\n")

w("## How the images were chosen\n")
w("**Tortoises.** Research-grade iNaturalist observations of *Gopherus berlandieri* (wild, from "
  "Texas and northern Mexico) whose photo carries a CC0, CC BY, CC BY-NC, CC BY-SA or CC BY-NC-SA "
  "licence. No-derivatives licences were left out so the images can be resized and augmented. "
  "Observations annotated as dead, or as bones or tracks rather than the animal, were dropped. One "
  "photo per observation, at iNaturalist's \"large\" size (1,024 pixels on the long side). The paper "
  "also drew some of its tortoise images from iNaturalist.\n")
w("**Blank frames.** Trail-camera frames with no animal: 156 from the South Texas sample that "
  "Kutugata et al. published with their nilgai model (the \"South Texas model\" the paper compares "
  "against), and the rest from NACTI's cameras at Archbold, Florida, where cattle are the most "
  "common animal in the frames. At most one frame per trigger burst.\n")
w("**Other animals.** As the paper did, armadillos get extra weight because they are the animal "
  "most likely to be mistaken for a tortoise. Half the armadillos are iNaturalist photos taken "
  "inside a box around the paper's four study sites; the other half are NACTI trail-camera frames. "
  "The rest are trail-camera frames of animals that also live in South Texas: raccoon, feral hog, "
  "deer, coyote, bobcat, cattle, birds, opossum, rabbit and wild turkey from the Florida site, and "
  "nilgai, cattle, birds, deer, feral hogs, coyote and raccoon from the South Texas sample. At most one "
  "frame per trigger burst and three per camera per day.\n")
w("**Camera strips trimmed.** Every NACTI frame came from a Reconyx camera that stamps a black "
  "strip with date and temperature along the top and the camera number and logo along the bottom. "
  "The tortoise photos have no such strips, so a model could have learned \"black strip means no "
  "tortoise\". The top 36 and bottom 64 pixels of each NACTI frame were cut off, leaving "
  "2048 x 1436 pixels, and the frame was re-saved as JPEG at quality 95. The South Texas frames "
  "arrived already without strips (669 x 346 pixels). The `image_url` column still points to the "
  "untrimmed originals.\n")
w("**Training versus hold-out.** The paper split its images at random and says this let frames "
  "from one tortoise sequence land on both sides. Here the split is by source unit instead: no "
  "iNaturalist photographer and no NACTI camera has images in both `train/` and `test/`. The South "
  "Texas sample has no camera or sequence information, so its images were split at random. The "
  "hold-out set draws from every source in roughly the same proportions as the training set, as the "
  "paper's did.\n")

w("## How this differs from the paper's images\n")
w("The paper's tortoise images were mainly frames from its own trail cameras: Browning Dark "
  "Ops Pro DCL cameras about 40 cm off the ground, tilted about 45 degrees down, taking a picture "
  "every two minutes in daylight, at four South Texas sites in 2022 and 2023. It topped these up with "
  "iNaturalist photos and Reconyx photos from highway wildlife crossings, but gives no count per "
  "source. None of its trail-camera frames are public, and iNaturalist has almost no camera-trap "
  "pictures of this species (5 of its roughly 6,000 Texas tortoise observations even mention a "
  "camera), so every tortoise here is a hand-held photo, often a close-up. The blanks and most other animals, by "
  "contrast, are trail-camera frames. A model trained on this set can therefore learn \"hand-held "
  "photo means tortoise, trail-camera frame means no tortoise\". The iNaturalist armadillos are "
  "there to blunt that, but they do not remove it. Scores on the hold-out set will be higher than "
  "the same model would earn on real trail-camera footage. To measure that, you need the authors' "
  "frames.\n")
w("Other differences: the NACTI metadata has no capture times, so night-time frames could not be "
  "left out and some are included, while the paper's time-lapse pictures were all taken in daylight. The labels are the source "
  "datasets' own, so a few NACTI \"empty\" frames may hold an animal and a few animal frames may "
  "show it only partly. The paper also gave the habitat mix of its images (Table 5: Tamaulipan "
  "thornscrub 678, brush 210, grassland/prairie 465, bare ground 137). The public sources record no "
  "habitat, so that mix is not reproduced. The paper's training images carried hand-drawn "
  "tortoise outlines (CVAT polygons, COCO format). This set has image-level labels only, with no "
  "boxes or outlines. Table 1 does not say whether M4's 2,705 training images include augmented "
  "copies. Here all 2,705 are distinct originals.\n")

w("## Licences and credit\n")
w("| Licence | Images |")
w("|---|---:|")
for lic, n in C(r["license"] for r in rows).most_common():
    w(f"| {lic} | {n:,} |")
w("")
w("iNaturalist photos keep their photographers' Creative Commons licences. The `attribution` "
  "column holds each credit line. Many are non-commercial (NC), so this set is for research use. "
  "NACTI is released under the Community Data License Agreement (permissive); cite Tabak MA et al. "
  "2019, \"Machine learning to classify animal species in camera trap images: Applications in "
  "ecology\", *Methods in Ecology and Evolution* 10(4): 585-590. The South Texas sample is MIT "
  "licensed, copyright 2020 Matthew Kutugata (github.com/mkutu/Nilgai); see Kutugata et al. 2021, "
  "*Journal of Fish and Wildlife Management* 12(2): 412-421.\n")

open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L[:40]))
