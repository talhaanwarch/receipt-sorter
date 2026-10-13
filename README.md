# receipt-sorter

Reads a receipt photo with an open-weight vision model and returns the company, the date and
the total as JSON, or as a CSV for your spreadsheet. Built for the DEV Hacktoberfest
Open-Source AI Challenge, week 2. One prompt in [AcruxCore](https://github.com/AcruxCore/AcruxCore)
holds the image and the instructions; the gateway calls the model and records a trace.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install acruxcore==0.17.0 streamlit python-dotenv pandas
cp .env.example .env                # fill in your AcruxCore key and URL
python scripts/setup_acruxcore.py   # registers models, creates the prompt and the dataset
streamlit run app.py
```

## Compare models

```bash
python scripts/run_eval.py --repeat 3
python scripts/score_run.py <run-id> ...
```

## Test set

50 receipts from the test split of [ICDAR 2019 SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE)
(CC BY 4.0, see `NOTICE`). The images are in `data/receipts/` and the dataset links to them on
`raw.githubusercontent.com`, so the model's provider can download them. `python tools/build_sample.py`
rebuilds the same 50 receipts and `data/labels.json` (needs `pyarrow` and `pillow`).

Code: Apache-2.0.
