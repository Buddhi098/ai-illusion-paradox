# app.py
import streamlit as st
import numpy as np
import joblib, json, os
from src.features import extract_features, FEATURE_NAMES
from src.train import train_model  # corrected import if file is train_model.py
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Human-or-AI Detector", layout="centered")
st.title("🤖 Human-or-AI Probability App")

# -----------------------------
# Section 1: Train Model Button
# -----------------------------
st.subheader("📊 Model Training")

if st.button("Train Model"):
    try:
        metrics = train_model("data/balanced_ai_human_prompts.csv")
        st.success("✅ Model trained and saved successfully!")
        st.json(metrics)
    except FileNotFoundError:
        st.error("⚠️ Missing dataset: 'data/text_samples.csv' not found.")


# -----------------------------
# Section 2: Load Model
# -----------------------------
if all(os.path.exists(f) for f in [
    "models/model.joblib",
    "models/scaler.joblib",
    "models/feature_list.json"
]):
    model = joblib.load("models/model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    with open("models/feature_list.json") as f:
        feat_names = json.load(f)
else:
    st.warning("Please train the model first before analysis.")
    st.stop()


# -----------------------------
# Section 3: Text Analysis
# -----------------------------
st.subheader("🧩 Analyze Text")

txt = st.text_area("Paste 50–250 words of text:", height=200)

word_count = len(txt.split())
st.write(f"📝 Word count: {word_count}")

if st.button("Analyze") and txt.strip():
    # Extract features
    x = extract_features(txt)
    xs = scaler.transform([x])[0]

    # Predict probability
    p_human = float(model.predict_proba([xs])[0, 1])
    st.metric("P(Human)", f"{p_human * 100:.1f}%")

    # -----------------------------
    # Top Feature Contribution
    # -----------------------------
    st.subheader("Top Contributing Features")

    coef = None
    base = None

    # Check if the model is an ensemble or a calibrated model
    if hasattr(model, "base_estimator_") or hasattr(model, "estimators_"):
        # Get the base estimator
        if hasattr(model, "base_estimator_"):
            base = model.base_estimator_
        else:  # model has "estimators_"
            base = model.estimators_[0]

        # If base is LogisticRegression, get coefficients directly
        if isinstance(base, RandomForestClassifier):
            coef = base.coef_[0]
        # If base has its own estimators (e.g., calibration with cv>1)
        elif hasattr(base, "estimators_") and len(base.estimators_) > 0:
            first_est = base.estimators_[0]
            if hasattr(first_est, "coef_"):
                coef = first_est.coef_[0]

    # coef now holds the coefficients if found, else None


    # Case 2: Plain logistic regression
    elif hasattr(model, "coef_"):
        coef = model.coef_[0]

    if coef is not None:
        contrib = np.abs(coef * xs)
        top_idx = np.argsort(contrib)[-3:][::-1]
        for i in top_idx:
            st.write(f"• **{feat_names[i]}** (scaled value {xs[i]:+.2f})")
    else:
        st.info("This model type does not provide interpretable feature coefficients.")

    st.caption("Note: Probabilistic estimate. Use with human review. Calibrated on a small dataset.")
