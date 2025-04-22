import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ─── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Automotive ML Dashboard",
    layout="wide"
)

DATA_DIR    = "data/processed"    # adjust to your JSONL folder
MODEL_PATH  = "models/model.pkl"     # adjust to your trained pipeline

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def load_jsonl_safe(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

@st.cache_data
def load_review_data():
    records, title_map = [], {}
    for split in ("train","val","test"):
        for itm in load_jsonl_safe(f"{DATA_DIR}/{split}.jsonl"):
            rev  = itm.get("review",{})
            meta = itm.get("meta",{})
            asin   = rev.get("asin")
            rating = rev.get("rating")
            title  = meta.get("title","")
            if asin and rating is not None:
                records.append({"asin":asin,"rating":int(rating)})
                title_map[asin] = title
    df = pd.DataFrame(records)
    top20 = df["asin"].value_counts().nlargest(20).index.tolist()
    return df[df["asin"].isin(top20)], top20, title_map

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except:
        st.error(f"Could not load model at `{MODEL_PATH}`.")
        st.stop()

# ─── MAIN LAYOUT ────────────────────────────────────────────────────────────────
st.title("Automotive ML Dashboard")

# Load shared resources once
df, top_asins, title_map = load_review_data()
model = load_model()

# ─── SIDE BY SIDE SECTIONS ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)

# — Section 1: Historical Review Analyzer ——————————————————————————
with col1:
    st.header("Review Rating Distribution")
    sel_asin = st.selectbox("Choose ASIN", top_asins)
    hist_data = (
        df[df["asin"] == sel_asin]["rating"]
        .value_counts()
        .reindex([1,2,3,4,5], fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(5,3))
    hist_data.plot.bar(ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title(f"{sel_asin}")
    ax.grid(axis="y")
    st.pyplot(fig)
    st.markdown(f"**Title:** {title_map.get(sel_asin,'—')}")

# — Section 2: Description → Rating Predictor ——————————————————————
with col2:
    st.header("Predict Customer Response from Description")
    desc = st.text_area("Product Description", height=250)
    if st.button("Predict Rating"):
        if not desc.strip():
            st.error("❗ Enter a description first.")
        else:
            try:
                pred = model.predict([desc])[0]
                st.success(f"⭐ Predicted Rating: {pred:.2f}")
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([desc])[0]
                    st.subheader("Rating Probabilities")
                    for star, p in enumerate(probs, start=1):
                        st.write(f"- {star} stars: {p:.1%}")
            except Exception as e:
                st.error("🚨 Prediction failed.")
                st.exception(e)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2025 Automotive ML Dashboard • TF‑IDF + Ridge model") 
