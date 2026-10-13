"""Receipt sorter: drop receipt photos, get a CSV of company, date and total.

    streamlit run app.py

Each photo goes through the AcruxCore gateway with the versioned `receipt-extract`
prompt, so every receipt leaves a trace you can open.
"""
import asyncio
import io

import pandas as pd
import streamlit as st

from receipt_sorter import config
from receipt_sorter.extract import extract_fields, to_data_url

MODELS = {
    "Gemma 4 31B (open)": "rs-gemma-4-31b",
    "Qwen3-VL 30B (open)": "rs-qwen3-vl-30b",
    "Qwen3-VL 8B (open, laptop-sized)": "rs-qwen3-vl-8b",
}

st.set_page_config(page_title="Receipt sorter", page_icon="🧾", layout="centered")
st.title("Receipt sorter")
st.caption("Reads receipt photos with an open-weight vision model and gives you a CSV for your spreadsheet.")

with st.sidebar:
    model_label = st.selectbox("Model", list(MODELS))
    alias = st.selectbox("Prompt version", ["production", "staging"], help="The alias of receipt-extract in AcruxCore.")

files = st.file_uploader("Receipt photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if files and st.button("Extract", type="primary"):
    rows, bar = [], st.progress(0.0)
    for i, upload in enumerate(files, 1):
        try:
            fields = asyncio.run(extract_fields(to_data_url(upload.getvalue(), upload.name), MODELS[model_label], alias, tags=["app"]))
            trace = f"{config.ACRUXCORE_DASHBOARD_URL}/traces/{fields.trace_id}" if fields.trace_id else ""
            rows.append({"file": upload.name, "company": fields.company, "date": fields.date, "total": fields.total, "seconds": round(fields.seconds, 1), "trace": trace})
        except Exception as exc:  # show the failure next to the file instead of losing the batch
            rows.append({"file": upload.name, "company": "", "date": "", "total": "", "seconds": 0, "trace": f"error: {exc}"})
        bar.progress(i / len(files))
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, column_config={"trace": st.column_config.LinkColumn("AcruxCore trace")})
    buffer = io.StringIO()
    table.drop(columns=["trace"]).to_csv(buffer, index=False)
    st.download_button("Download CSV", buffer.getvalue(), "receipts.csv", "text/csv")
