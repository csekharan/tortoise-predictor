# Texas tortoise composite image set

This folder holds public images arranged to match the training and hold-out image counts that Tleimat, Krell and McCracken (2026) report for their Texas tortoise detector (Table 1 of "Needle in a haystack: towards the use of machine learning to detect Texas tortoises in large datasets", *Amphibian & Reptile Conservation* 20(1): 67-81). The paper's own images were never published, so these are different images chosen to fill the same slots: the same number of tortoise, no-animal and other-animal pictures in the training set (the paper's model M4, the one released on GitHub) and in the hold-out test set.

## Composition

| Split | Tortoise | No animal (`blank/`) | Other animal | Total | Paper (Table 1) |
|---|---:|---:|---:|---:|---|
| Training | 1,332 | 420 | 953 | 2,705 | 1,332 / 420 / 953 = 2,705 |
| Hold-out test | 165 | 65 | 44 | 274 | 165 / 65 / 44 = 274 |
| All | 1,497 | 485 | 997 | 2,979 | |

## Hold-out test set

Every hold-out image is a photo, not a trail-camera frame, and every one was looked at before any model was run on it.

| Class | Group | Images | Source | What it shows |
|---|---|---:|---|---|
| `tortoise` | Texas tortoise | 165 | iNaturalist photos (hand-held) | *Gopherus berlandieri*, wild, Texas and northern Mexico |
| `blank` | No animal: vegetation and bushes | 22 | iNaturalist photos (hand-held) | plants photographed around the paper's four study sites |
| `blank` | No animal: dry land | 21 | Wikimedia Commons photos | cracked mud, caliche, gravel, bare soil, desert floor, dry creek beds |
| `blank` | No animal: beaches | 22 | Wikimedia Commons photos | sand, dunes, seaweed and wrack, driftwood, surf |
| `other_animal` | Armadillos | 11 | iNaturalist photos (hand-held) | nine-banded armadillos photographed around the paper's four study sites |
| `other_animal` | Pangolins | 11 | iNaturalist photos (hand-held) | Africa and Asia |
| `other_animal` | Crabs | 11 | iNaturalist photos (hand-held) | true crabs photographed in Texas |
| `other_animal` | Other sea creatures | 11 | iNaturalist photos (hand-held) | Atlantic Horseshoe Crab (3), Purple-brown Urchin (2), Two-banded Sea Star (2), Woody Chiton (1), Texas Lightning Whelk (1), Portuguese Man o' War (1), Cannonball Jelly (1) |

- **Pangolins:** Temminck's Ground Pangolin (4), Black-bellied Pangolin (3), White-bellied Pangolin (2), Taiwan Pangolin (1), Sunda Pangolin (1).
- **Crabs:** Atlantic Blue Crab (6), Atlantic Ghost Crab (3), Speckled Swimming Crab (1), Gulf Marsh Fiddler Crab (1).

iNaturalist now files the Texas nine-banded armadillo as *Dasypus mexicanus*, "Mexican long-nosed armadillo", and that is the name in the `species` column.

**How the hold-out images were chosen.** `build_holdout.py OUT_DIR CACHE_DIR` (Python, standard library only) rebuilds the hold-out set:

- **Tortoises.** The tortoise photos are research-grade iNaturalist observations of *Gopherus berlandieri*, chosen as described under the training set below. An earlier version of this hold-out set had one photo labelled tortoise that shows a bulldozer and no tortoise; it was replaced with another tortoise photo drawn the same way.
- **No animal: vegetation and bushes.** Research-grade iNaturalist plant observations from the box around the paper's four study sites, chosen like the animal photos below. These are hand-held photos of the plants a Texas tortoise lives among, taken the way the tortoise photos were.
- **No animal: dry land and beaches.** Wikimedia Commons photos found with fixed search phrases for each group (the phrases are listed in the script), JPEG, at least 1,000 by 650 pixels, under CC0, public-domain, CC BY or CC BY-SA licences. Titles that name an animal, a person, a building, a vehicle, a map or a document were skipped. Downloaded at 960 pixels wide.
- **Other animals.** Research-grade iNaturalist observations, wild, with a CC0, CC BY, CC BY-NC, CC BY-SA or CC BY-NC-SA photo, not annotated as dead or as bones, tracks or moults. Armadillos come from the box around the paper's four study sites, crabs and most sea creatures from Texas, and pangolins, horseshoe crabs and chitons from anywhere.
- **Order and checks.** Each group's candidates are shuffled with a fixed seed and taken in that order, at most one image per photographer or Commons author in each group. No hold-out image comes from an iNaturalist photographer who has images in `train/`. Each picked image was then looked at, and the next candidate took the place of any that broke these rules:
  - A no-animal photo must show no animal at all: no birds in the sky, insects on flowers, snails on seaweed, people, hands or feet. It must show the kind of ground its group names, and no building, vehicle, sign, document, drawing or studio backdrop may be its subject.
  - An animal photo must show the animal itself, not a burrow, tracks, a carcass, a bleached skeleton (most Texas sand-dollar photos are skeletons) or a trail-camera frame, and no other animal may be the main subject.

`REJECTED` in the script lists all 162 rejected candidates, 128 no-animal photos and 34 animal photos, each with its reason.

The models played no part in choosing images, with one exception. After the first runs, every image that either method got wrong, or that Gemini described differently from its label, got a second, closer look. Three broke the rules and were replaced by the next candidates: a beach photo with a sea-turtle hatchling in the wrack, a beach photo with what may be a grasshopper, and a worn, empty whelk shell. Their entries in `REJECTED` say so.

## Training set

The training set is the one built for the paper's model M4 and is not used by the comparison in this repository; it is there for retraining or few-shot prompting. `build_tortoise_dataset.py OUT_DIR CACHE_DIR` downloads it, and `crop_camera_strips.ps1` then trims the camera strips described below. (That script also built the first version of the hold-out set, which `build_holdout.py` replaced.)

| Class | Source | Images |
|---|---|---:|
| tortoise | iNaturalist photos (hand-held) | 1,332 |
| blank | Nilgai-study trail cameras, South Texas | 135 |
| blank | NACTI trail cameras, Archbold, Florida | 285 |
| other_animal | Nilgai-study trail cameras, South Texas | 62 |
| other_animal | NACTI trail cameras, Archbold, Florida | 724 |
| other_animal | iNaturalist photos (hand-held) | 167 |

The training set's other-animal images, by species:

| Species | Images | Hand-held photos | Trail-camera frames |
|---|---:|---:|---:|
| nine-banded armadillo | 334 | 167 | 167 |
| cattle | 76 | 0 | 76 |
| bird | 71 | 0 | 71 |
| deer | 64 | 0 | 64 |
| coyote | 63 | 0 | 63 |
| feral hog | 63 | 0 | 63 |
| raccoon | 62 | 0 | 62 |
| bobcat | 61 | 0 | 61 |
| opossum | 48 | 0 | 48 |
| rabbit | 42 | 0 | 42 |
| wild turkey | 38 | 0 | 38 |
| nilgai | 31 | 0 | 31 |

**Tortoises.** Research-grade iNaturalist observations of *Gopherus berlandieri* (wild, from Texas and northern Mexico) whose photo carries a CC0, CC BY, CC BY-NC, CC BY-SA or CC BY-NC-SA licence. No-derivatives licences were left out so the images can be resized and augmented. Observations annotated as dead, or as bones or tracks rather than the animal, were dropped. One photo per observation, at iNaturalist's "large" size (1,024 pixels on the long side). The paper also drew some of its tortoise images from iNaturalist.

**Blank frames.** Trail-camera frames labelled empty by their source: from the South Texas sample that Kutugata et al. published with their nilgai model (the "South Texas model" the paper compares against), and from NACTI's cameras at Archbold, Florida. At most one frame per trigger burst. These labels were not checked by eye, and a few frames do show an animal.

**Other animals.** As the paper did, armadillos get extra weight because they are the animal most likely to be mistaken for a tortoise. Half the armadillos are iNaturalist photos taken inside a box around the paper's four study sites; the other half are NACTI trail-camera frames. The rest are trail-camera frames of animals that also live in South Texas. At most one frame per trigger burst and three per camera per day.

**Camera strips trimmed.** Every NACTI frame came from a Reconyx camera that stamps a black strip with date and temperature along the top and the camera number and logo along the bottom. The tortoise photos have no such strips, so a model could have learned "black strip means no tortoise". The top 36 and bottom 64 pixels of each NACTI frame were cut off, leaving 2048 x 1436 pixels, and the frame was re-saved as JPEG at quality 95. The South Texas frames arrived already without strips (669 x 346 pixels). The `image_url` column still points to the untrimmed originals.

## Folder layout

`train/` and `test/` each contain `tortoise/`, `blank/` (no animal) and `other_animal/`. That is the one-folder-per-class layout that PyTorch `ImageFolder` and Keras `image_dataset_from_directory` read directly. `manifest.csv` has one row per image: split, class, species, file path, source, whether it is a trail-camera frame, the photographer, camera or Commons author it came from (`group`), the original URL and page, licence, attribution, place, date and the group within its class (`stratum`).

## How this differs from the paper's images

The paper's tortoise images were mainly frames from its own trail cameras: Browning Dark Ops Pro DCL cameras about 40 cm off the ground, tilted about 45 degrees down, taking a picture every two minutes in daylight, at four South Texas sites in 2022 and 2023. It topped these up with iNaturalist photos and Reconyx photos from highway wildlife crossings, but gives no count per source. None of its trail-camera frames are public, and iNaturalist has almost no camera-trap pictures of this species, so every tortoise here is a hand-held photo, often a close-up. In the hold-out set every other image is a photo too, so a model cannot tell the classes apart by camera type there. In the training set the blanks and most other animals are trail-camera frames, so a model trained on it could learn "hand-held photo means tortoise". Scores on the hold-out set will be higher than the same model would earn on real trail-camera footage. To measure that, you need the authors' frames.

The paper's negatives were trail-camera frames from its own sites. The hold-out negatives here are chosen to be hard in a different way: animals with a hard shell or a domed back, and empty scenes of the ground a tortoise walks on, including Texas beaches. The paper also gave the habitat mix of its images (Table 5: Tamaulipan thornscrub 678, brush 210, grassland/prairie 465, bare ground 137); that mix is not reproduced. The paper's training images carried hand-drawn tortoise outlines (CVAT polygons, COCO format). This set has image-level labels only, with no boxes or outlines.

## Licences and credit

| Licence | Images |
|---|---:|
| CC-BY-NC | 1,576 |
| CDLA-Permissive-1.0 | 1,009 |
| MIT | 197 |
| CC-BY | 99 |
| CC0 | 23 |
| CC-BY-NC-SA | 20 |
| CC-BY-SA | 14 |
| CC-BY-SA-4.0 | 13 |
| CC-BY-SA-2.0 | 9 |
| Public domain | 6 |
| CC-BY-2.0 | 3 |
| CC-BY-SA-3.0 | 3 |
| CC-BY-SA-2.5 | 2 |
| CC-BY-4.0 | 2 |
| CC-BY-2.5 | 1 |
| CC-BY-3.0 | 1 |
| CC-BY-3.0-US | 1 |

iNaturalist photos keep their photographers' Creative Commons licences, and Wikimedia Commons photos their authors' licences. The `attribution` column holds each credit line. Many iNaturalist photos are non-commercial (NC), so this set is for research use. NACTI is released under the Community Data License Agreement (permissive); cite Tabak MA et al. 2019, "Machine learning to classify animal species in camera trap images: Applications in ecology", *Methods in Ecology and Evolution* 10(4): 585-590. The South Texas sample is MIT licensed, copyright 2020 Matthew Kutugata (github.com/mkutu/Nilgai); see Kutugata et al. 2021, *Journal of Fish and Wildlife Management* 12(2): 412-421.
