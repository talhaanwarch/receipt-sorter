# receipt-sorter

Reads Arabic, English and mixed-language receipts with an open-weight vision model
and writes a CSV (store, date, total). Built for the DEV Hacktoberfest Open-Source AI
Challenge, week 2. Model calls go through [AcruxCore](https://github.com/AcruxCore/AcruxCore),
which versions the prompt and scores each model against a labelled test set.

Work in progress. The test set comes from the MIT-licensed
[ReceiptSense (CORU)](https://huggingface.co/datasets/abdoelsayed/CORU) dataset.
