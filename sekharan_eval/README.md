# sekharan_eval

Code for the comparison described in the [main README](../README.md): zero-shot Gemini Flash against the original Mask R-CNN, on the 274 hold-out images in `../data/texas_tortoise_composite/test/`. Nothing here imports the original code; `eval_maskrcnn.py` only reads its weights, `../tleimat_original/tortoisemodel.keras`.

| Script | What it does | Needs |
|---|---|---|
| `zero_shot_gemini.py` | Asks Gemini, with no training and no examples, whether each image shows an animal and which one. An image counts as a tortoise when the answer names a tortoise or turtle. | Python 3.10+, `GEMINI_API_KEY` |
| `eval_maskrcnn.py` | Runs the original model on the same images. An image counts as a tortoise when any detection scores above `--threshold` (default 0.5). | `requirements.txt` |
| `compare_models.py` | Joins the two prediction files. Prints a breakdown by image type, where the models agree, misses, top false alarms and the threshold sweep. | Python only |
| `make_figures.py` | Draws the README figures, in light and dark versions, into `figures/`. Prints the numbers the README quotes. | Python only |

```text
python zero_shot_gemini.py                    # original prompt
python zero_shot_gemini.py --prompt neutral   # names no animals, asks no tortoise question
python zero_shot_gemini.py --limit 10         # quick smoke test across classes
python eval_maskrcnn.py                       # about 14 min on a laptop CPU
python eval_maskrcnn.py --threshold 0.85      # headline metrics at another threshold
python compare_models.py                      # or: compare_models.py GEMINI_CSV MASKRCNN_CSV MASKRCNN_JSON
python make_figures.py
```

Results go to `results/`:
- `predictions_*.csv`: one row per image.
- `metrics_*.json`: totals, plus Mask R-CNN's threshold sweep.
- The neutral prompt writes `predictions_gemini-3.8-flash_neutral.csv` and `metrics_gemini-3.8-flash_neutral.json`.

Both scorers save as they go, so an interrupted run resumes where it stopped. Use `--fresh` with `eval_maskrcnn.py` to start over.

The CSV and JSON files keep the manifest's labels. `make_figures.py` corrects one of them: it counts the bulldozer photo (`test/tortoise/inat-tortoise_inat50015197_p79414426.jpg`) as not a tortoise.
