# Finding Texas tortoises in images: a trained Mask R-CNN vs. zero-shot Gemini

Texas tortoises (*Gopherus berlandieri*) hide in thick brush and are hard to survey by eye, so researchers set time-lapse camera traps that produce huge numbers of frames, almost none of which show a tortoise. Tleimat, Krell and McCracken (2026) trained a Mask R-CNN to find tortoises in those frames and released it.

This repository asks whether a general-purpose multimodal model can do the same job **with no training at all**. It runs the released Mask R-CNN and Gemini 3.8 Flash on the same 274 hold-out images and compares the results.

## At a glance

| | Mask R-CNN, trained | Gemini, original prompt | Gemini, neutral prompt |
|---|:---:|:---:|:---:|
| **Tortoises found** (of 164) | **164** | **164** | **164** |
| **False alarms** (of 110 images without a tortoise) | 18 at threshold 0.5<br/>7 at 0.85 | 1 | **0** |
| **Species named correctly** | does not name species | 152 (93%) | 60 (37%) |
| **Training images needed** | 2,705, every tortoise outlined by hand | none | none |
| **Time for 274 images** | 14 min, laptop CPU | 5.6 min, API | 7.2 min, API |
| **Runs offline** | yes | no | no |

All three find every tortoise; they differ in false alarms. The original prompt lists example animals, Texas tortoise among them, and the neutral prompt names none. Dropping the hint removed Gemini's one false alarm, but most of its correct species names went with it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="sekharan_eval/figures/confusion-dark.svg">
  <img alt="Confusion matrices for 164 tortoise and 110 other images. Gemini original prompt: 164 found, 0 missed, 1 false alarm, 109 correct. Gemini neutral prompt: 164 found, 0 missed, 0 false alarms, 110 correct. Mask R-CNN at threshold 0.5: 164 found, 0 missed, 18 false alarms, 92 correct. Mask R-CNN at threshold 0.85: 164 found, 0 missed, 7 false alarms, 103 correct." src="sekharan_eval/figures/confusion-light.svg">
</picture>

## The two methods

```mermaid
flowchart LR
    img["Image<br/>camera trap or phone"]
    subgraph og ["Original: trained Mask R-CNN (Tleimat et al. 2026)"]
        direction LR
        m1["Mask R-CNN, ResNet-50 FPN<br/>fine-tuned on 2,705 labelled images"] --> m2["Tortoise outlines<br/>with confidence scores"] --> m3{"Any score<br/>above 0.5?"}
    end
    subgraph new ["New: zero-shot Gemini 3.8 Flash"]
        direction LR
        g1["Image plus one short prompt<br/>no training, no examples"] --> g2["JSON answer<br/>is there an animal, which one"] --> g3{"Names a tortoise<br/>or turtle?"}
    end
    img --> m1
    img --> g1
    m3 -- yes --> t(["Tortoise"])
    g3 -- yes --> t
```

| | Original: Mask R-CNN | New: zero-shot Gemini |
|---|---|---|
| **Model** | Mask R-CNN with a ResNet-50 FPN backbone, pretrained on COCO and fine-tuned for one class, "tortoise" | Gemini 3.8 Flash, a general multimodal model, used as is |
| **What it needs** | 2,705 training images with every tortoise outlined by hand | Nothing: no training and no example images |
| **Output per image** | Tortoise outlines and boxes, each with a confidence score | A JSON answer: is there an animal, and what is it |
| **Called a tortoise when** | Any detection scores above a threshold (0.5 by default) | The answer names a tortoise or turtle |
| **Runs on** | Your own computer, offline | Google's API, with an internet connection and an API key |

Gemini gets one of two prompts with each image, at temperature 0, and must answer in a fixed JSON format:

<table>
<tr><th width="50%">Original prompt</th><th width="50%">Neutral prompt</th></tr>
<tr>
<td>

```text
Look at this photograph. Is there an animal
in it?
If yes, give the animal's common name as
precisely as you can (for example: Texas
tortoise, box turtle, nine-banded armadillo,
raccoon, white-tailed deer, cattle, bird,
feral hog, coyote).
If there is no animal, answer no animal.
Return JSON only.
```

</td>
<td>

```text
Look at this photograph. Is there an animal
in it?
If yes, give the animal's common name as
precisely as you can.
If there is no animal, answer none.
Return JSON only.
```

</td>
</tr>
<tr>
<td>Answer fields: <code>animal_present</code>, <code>animal</code>, <code>is_tortoise_or_turtle</code></td>
<td>Answer fields: <code>animal_present</code>, <code>animal</code>. The tortoise call comes from the name alone.</td>
</tr>
</table>

## The images

The paper's images were never published. [`data/texas_tortoise_composite/`](data/texas_tortoise_composite/) rebuilds the paper's Table 1 from public sources, with the same number of tortoise, empty and other-animal images in each split:

| Split | Tortoise | Empty frame | Other animal | Total |
|---|---:|---:|---:|---:|
| Training | 1,332 | 420 | 953 | 2,705 |
| Hold-out | 165 | 65 | 44 | 274 |

- **Tortoises:** research-grade iNaturalist photos.
- **Empty frames and most other animals:** trail-camera frames, from NACTI's Florida cameras and from the South Texas sample published with a nilgai detection study.
- **Armadillos:** weighted heavily, as in the paper, because they look most like tortoises.

No iNaturalist photographer and no NACTI camera appears in both splits. This comparison uses only the hold-out images. The training split is there for anyone who wants to retrain the detector or try few-shot prompting.

One hold-out photo labeled tortoise (`inat-tortoise_inat50015197_p79414426.jpg`) shows a bulldozer and a pickup truck and no tortoise. The results below count it as not a tortoise, which leaves 164 tortoise images and 110 others.

## Results

| Method | Found | Missed | False alarms | Precision | Recall | F1 | Accuracy | False-alarm rate (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Gemini 3.8 Flash, neutral prompt** | **164** | **0** | **0** | **1.000** | **1.000** | **1.000** | **1.000** | **0.0%** (0.0–3.3%) |
| Gemini 3.8 Flash, original prompt | 164 | 0 | 1 | 0.994 | 1.000 | 0.997 | 0.996 | 0.9% (0.0–5.0%) |
| Mask R-CNN, threshold 0.5 (paper's default) | 164 | 0 | 18 | 0.901 | 1.000 | 0.948 | 0.934 | 16.4% (10.0–24.6%) |
| Mask R-CNN, threshold 0.85 (paper's best F1) | 164 | 0 | 7 | 0.959 | 1.000 | 0.979 | 0.974 | 6.4% (2.6–12.7%) |
| Mask R-CNN, threshold 0.95 | 161 | 3 | 4 | 0.976 | 0.982 | 0.979 | 0.974 | 3.6% (1.0–9.0%) |

Intervals are exact Clopper–Pearson intervals. The two methods made their false alarms on different images, so the gap is unlikely to be chance (exact McNemar test on the 110 non-tortoise images):
- **Mask R-CNN vs. the original prompt:** 18 images only Mask R-CNN flagged, 1 only Gemini flagged (p < 0.001).
- **Mask R-CNN vs. the neutral prompt:** 18 against 0 (p < 0.0001).

For context, the paper reports this same model on its own trail-camera hold-out set, at threshold 0.5 and counting each tortoise whose box overlaps the drawn outline: precision 0.93, recall 0.83, accuracy 0.86, false-positive rate 0.09 and F1 0.88 (Table 3). Recall is much higher here because every tortoise in this hold-out set is a clear hand-held photo; see [Limits](#limits).

### Where the false alarms come from

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="sekharan_eval/figures/false-alarms-dark.svg">
  <img alt="Images without a tortoise that each method called tortoise. Mask R-CNN: 6 of 16 armadillos, 3 of 3 feral hogs, 2 of 21 South Texas empty frames, 2 of 4 deer, 2 of 2 wild turkeys, 1 bobcat, 1 bird and the bulldozer photo. Gemini with the original prompt: 1 South Texas empty frame. Gemini with the neutral prompt: none. No method flagged the 44 Florida empty frames or the 14 other animals." src="sekharan_eval/figures/false-alarms-light.svg">
</picture>

- **Armadillos:** they account for a third of Mask R-CNN's false alarms. Gemini called none of the 16 a tortoise under either prompt.
- **Other animals:** Mask R-CNN also flagged every feral hog and wild turkey in the set.
- **Gemini's one error:** with the original prompt, it called an empty South Texas trail-camera frame "Texas tortoise". With the neutral prompt it called the same frame "wild turkey": not a tortoise call, though the frame is labeled empty.
- **Empty Florida frames:** no method flagged any of the 44.

### Raising the threshold does not close the gap

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="sekharan_eval/figures/tradeoff-dark.svg">
  <img alt="Mask R-CNN's trade-off between tortoises found and false alarms across all thresholds, with Gemini's two points: neutral prompt at 0 false alarms and 164 found, original prompt at 1 and 164. Mask R-CNN at 0.5: 18 false alarms, 0 missed; at 0.85: 7 and 0; at 0.95: 4 and 3; at 0.99: 0 and 14." src="sekharan_eval/figures/tradeoff-light.svg">
</picture>

Mask R-CNN ranks images well (ROC AUC 0.997). Its trouble is a handful of confident mistakes:
- Every real tortoise scores at least 0.885, but six non-tortoise images score higher than that. The highest, an armadillo at 0.99, outscores 14 of the tortoises.
- At its best setting with no missed tortoises, it still raises 6 false alarms.
- Getting to zero false alarms costs 14 tortoises.

Gemini's two results sit in the top-left corner, where no point on the Mask R-CNN curve reaches.

### Does the prompt matter?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="sekharan_eval/figures/species-dark.svg">
  <img alt="What Gemini called the 164 tortoise photos. Original prompt: Texas tortoise 152, tortoise 5, desert tortoise 3, gopher tortoise 2, box turtle 2. Neutral prompt: Texas tortoise 60, tortoise 38, desert tortoise 26, gopher tortoise 22, red-footed tortoise 7, other tortoise species 9, box turtle 2." src="sekharan_eval/figures/species-light.svg">
</picture>

For finding tortoises, no: both prompts found all 164 tortoises. The neutral prompt even removed the one false alarm.

For naming the species, yes:

| What Gemini called the 164 Texas tortoises | Original prompt | Neutral prompt |
|---|---:|---:|
| Texas tortoise, the right species | **152** (93%) | 60 (37%) |
| "Tortoise", with no species | 5 | 38 |
| Desert or gopher tortoise, the two other North American *Gopherus* species | 5 | 48 |
| Species from other continents (red-footed, African spurred, Russian and others) | 0 | 16 |
| Box turtle | 2 | 2 |

Asked a yes-or-no tortoise question, Gemini does not need the hint. To name the species without it, the model probably needs other context, such as where the photo was taken; that is untested. The scoring here counts any tortoise or turtle answer as a hit. Box turtles also live in South Texas, so a real survey would need to treat "turtle" answers differently.

### Speed and cost

| | Mask R-CNN | Gemini, original prompt | Gemini, neutral prompt |
|---|---|---|---|
| **Where it ran** | Laptop CPU (Intel Core i7-1255U), PyTorch, offline | Gemini API | Gemini API |
| **Time per request** (median) | 3.0 s per image | 4.0 s | 5.4 s |
| **All 274 images** | 14 min 14 s | 5 min 34 s, 4 requests at a time | 7 min 10 s, 4 requests at a time |
| **Tokens per image** | none | about 1,420: 1,160 in, 30 out, 230 thinking | about 1,440: 1,130 in, 17 out, 300 thinking |

A GPU would make Mask R-CNN much faster; that was not measured here. Gemini's time depends on the network and on how many requests run at once. The neutral prompt is shorter, but Gemini spent more tokens thinking about each image.

## Limits

- **No trail-camera tortoises.** Every tortoise here is a hand-held photo, often a close-up, while the empty frames and most other animals are trail-camera frames. Hard cases from real deployments are untested for both methods: tortoises far away, partly hidden or cut off at the frame edge. Both methods could also partly be telling phone photos from camera frames.
- **Possible prior exposure.** The paper drew some of its training tortoises from iNaturalist, and Gemini was trained on web data, so either model may have seen some of these photos before.
- **Small negative set.** With 110 non-tortoise images, zero false alarms is consistent with a true rate of up to 3.3%, and one false alarm with up to 5%.
- **Per-image scoring.** An image counts as found if the method flags it anywhere. The paper's rule also requires the predicted box to overlap the drawn tortoise.

## Next steps

- Test both methods on real trail-camera tortoise frames, for example the authors' own.
- Add location to the neutral prompt (for example "taken in South Texas") and see whether species names recover without listing any species.
- Try few-shot prompts that use images from the training split.
- Try a cascade: Mask R-CNN screens every frame cheaply and offline, then Gemini checks only the frames it flags.

## Reproduce

```bash
git lfs install                    # the Mask R-CNN weights are stored with Git LFS
git clone https://github.com/csekharan/tortoise-predictor.git
cd tortoise-predictor/sekharan_eval
pip install -r requirements.txt    # torch, torchvision, pillow (CPU builds are enough)

python eval_maskrcnn.py                      # Mask R-CNN on the 274 hold-out images
export GEMINI_API_KEY=...                    # PowerShell: $env:GEMINI_API_KEY="..."
python zero_shot_gemini.py                   # Gemini, original prompt
python zero_shot_gemini.py --prompt neutral  # Gemini, neutral prompt
python compare_models.py                     # breakdown by image type
python make_figures.py                       # redraws the figures and prints the numbers above
```

Both scorers save as they go and resume where they stopped. See [`sekharan_eval/README.md`](sekharan_eval/README.md) for all options.

## Repository layout

```text
tortoise-predictor/
├── tleimat_original/            original detector code, weights and licence, unchanged
├── sekharan_eval/               this comparison; imports nothing from tleimat_original
│   ├── zero_shot_gemini.py      Gemini scorer, both prompts (standard library only)
│   ├── eval_maskrcnn.py         runs tleimat_original/tortoisemodel.keras on the same images
│   ├── compare_models.py        joins the prediction files
│   ├── make_figures.py          draws the figures in this README
│   ├── results/                 per-image predictions and metrics for every run
│   └── figures/
└── data/
    └── texas_tortoise_composite/   2,979 images, manifest.csv, build scripts; read by both
```

## Credits and licences

- **Original detector:** Tleimat JM, Krell E, McCracken SF. 2026. Needle in a haystack: towards the use of machine learning to detect Texas tortoises in large datasets. *Amphibian & Reptile Conservation* 20(1): 67–81 (e356). Code and weights from [github.com/jtleimat/tortoise-predictor](https://github.com/jtleimat/tortoise-predictor), MIT licence ([`tleimat_original/LICENSE`](tleimat_original/LICENSE)).
- **Images:** each image keeps its own licence. `manifest.csv` gives the licence, credit line and source page for every image.
  - iNaturalist photos are under their photographers' Creative Commons licences. About half are non-commercial, so the image set as a whole is for research use.
  - NACTI frames: Tabak MA et al. 2019, *Methods in Ecology and Evolution* 10(4): 585–590, via LILA BC, under the CDLA-Permissive-1.0 licence.
  - South Texas frames: Kutugata M et al. 2021, *Journal of Fish and Wildlife Management* 12(2): 412–421, MIT licence.

  See [`data/texas_tortoise_composite/README.md`](data/texas_tortoise_composite/README.md) for details.
