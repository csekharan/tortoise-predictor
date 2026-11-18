# sekharan_eval: zero-shot Gemini vs. the original Mask R-CNN

Scores Gemini Flash (zero-shot, no examples) and the original Tleimat et al. Mask R-CNN on the same 274 hold-out images in `../data/texas_tortoise_composite/test/`, and compares them.

| Script | What it does | Needs |
|---|---|---|
| `zero_shot_gemini.py` | Asks Gemini whether each image shows an animal and which one. An image counts as a tortoise when the answer names a tortoise or turtle. | Python 3.10+, `GEMINI_API_KEY` |
| `eval_maskrcnn.py` | Runs `../tleimat_original/tortoisemodel.keras` on the same images. An image counts as a tortoise when any detection scores above the threshold (default 0.5). | `requirements.txt` |
| `compare_models.py` | Joins the two prediction files and breaks the results down by true label and source. | Python only |

All three write to and read from `results/`. Both scorers resume an interrupted run.

```
python zero_shot_gemini.py                    # original prompt
python zero_shot_gemini.py --prompt neutral   # names no animals, asks no tortoise question
python eval_maskrcnn.py                       # about 14 min on a laptop CPU
python compare_models.py
```

## Results on the 274 hold-out images

| | TP / FN / FP / TN | Precision | Recall | False-positive rate |
|---|---|---|---|---|
| Gemini 3.8 Flash, zero-shot, original prompt | 164 / 0 / 1 / 109 | 0.994 | 1.000 | 0.009 |
| Mask R-CNN, threshold 0.5 | 164 / 0 / 18 / 92 | 0.901 | 1.000 | 0.164 |
| Mask R-CNN, threshold 0.85 | 164 / 0 / 7 / 103 | 0.959 | 1.000 | 0.064 |
| Mask R-CNN, threshold 0.95 | 161 / 3 / 4 / 106 | 0.976 | 0.982 | 0.036 |

These counts treat `test/tortoise/inat-tortoise_inat50015197_p79414426.jpg` as a negative. The manifest labels it a tortoise, but the photo shows a bulldozer and a pickup truck and no tortoise. The CSV and JSON files in `results/` keep the manifest's labels.

Mask R-CNN's false alarms at 0.5:
- 6 of 16 armadillos
- 3 of 3 feral hogs and 2 of 2 wild turkeys
- 2 of 4 deer, 1 bird and 1 bobcat
- 2 South Texas empty frames
- the pickup truck in the mislabeled photo

Gemini's one false alarm is a South Texas empty frame.

## Caveats

- All tortoise images are hand-held iNaturalist photos. There are no trail-camera tortoises.
- The paper trained on iNaturalist tortoise photos and Gemini was trained on web images, so either model may have seen some of these tortoise photos before.
- Scoring is per image, not the paper's rule that a predicted box must overlap the drawn tortoise.
