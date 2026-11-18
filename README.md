# Texas tortoise detection: original Mask R-CNN and zero-shot comparison

This private repository holds two separate code bases that share one data folder.

| Folder | What it is | Who wrote it |
|---|---|---|
| [tleimat_original/](tleimat_original/) | The Texas tortoise Mask R-CNN detector as released at [github.com/jtleimat/tortoise-predictor](https://github.com/jtleimat/tortoise-predictor), unchanged: code, model weights (`tortoisemodel.keras`), requirements and MIT licence. | J. Tleimat (Tleimat, Krell and McCracken 2026) |
| [sekharan_eval/](sekharan_eval/) | Zero-shot Gemini Flash scorer, an evaluation harness for the original model, the comparison script and the results. Has its own requirements. | Sekharan |
| [data/texas_tortoise_composite/](data/texas_tortoise_composite/) | Public images arranged to match the training and hold-out counts in the paper's Table 1, which both code bases read. | Built for this comparison |

No code is shared between the two code folders: `sekharan_eval` imports nothing from `tleimat_original`. Its Mask R-CNN harness only reads the original model weights, from `tleimat_original/tortoisemodel.keras`.

The images themselves (1.6 GB, `train/` and `test/`) are not in git. The manifest, README and build scripts in the data folder are, and `build_tortoise_dataset.py` downloads the same images again. See [data/texas_tortoise_composite/README.md](data/texas_tortoise_composite/README.md).

The model weights are stored with Git LFS, so clone with Git LFS installed.
