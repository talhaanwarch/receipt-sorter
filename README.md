# receipt-sorter

Reads scanned receipts with an open-weight vision model and writes a CSV (store, date, total). Built for the DEV Hacktoberfest Open-Source AI
Challenge, week 2. Model calls go through [AcruxCore](https://github.com/AcruxCore/AcruxCore),
which versions the prompt and scores each model against a labelled test set.

Work in progress.

## Test set

50 receipts from the test split of [ICDAR 2019 SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE)
(CC BY 4.0; Huang et al., ICDAR 2019 Robust Reading Challenge). The images are not in git.
Run `python tools/build_sample.py` to download the data and rebuild the same 50 receipts
and `data/labels.json`. Needs `pyarrow` and `pillow`.
