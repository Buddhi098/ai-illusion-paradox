# train_model.py
import pandas as pd
import numpy as np
import json, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from src.features import extract_features, FEATURE_NAMES

def dataset_preprocess(df):
    # Map 0/1 to 'human'/'ai'
    df['generated'] = df['generated'].map({0: 'human', 1: 'ai'})
    df.rename(columns={'generated': 'label'}, inplace=True)
    df = df.dropna(subset=['text', 'label'])
    return df

def train_model(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print("📘 Loading dataset...")
    df = pd.read_csv(data_path)  # expects columns: text, generated (0/1)

    df = dataset_preprocess(df)

    # -----------------------------
    # Extract features and labels
    # -----------------------------
    X = np.vstack([extract_features(t) for t in df["text"].astype(str)])
    y = (df["label"] == "human").astype(int).values  # 1 for human, 0 for AI

    # -----------------------------
    # Train / Validation split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    Xtr, Xv = scaler.transform(X_train), scaler.transform(X_val)

    # -----------------------------
    # Train Random Forest
    # -----------------------------
    print(f"🧮 Dataset size: {len(df)} samples")
    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(Xtr, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    probs = model.predict_proba(Xv)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "ROC-AUC": float(roc_auc_score(y_val, probs)),
        "Accuracy": float(accuracy_score(y_val, preds)),
        "Brier Score": float(brier_score_loss(y_val, probs)),
    }

    # -----------------------------
    # Save trained artifacts
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(model, "models/model.joblib")
    with open("models/feature_list.json", "w") as f:
        json.dump(FEATURE_NAMES, f)

    print("\n✅ Training complete! Saved:")
    print(" - models/scaler.joblib")
    print(" - models/model.joblib")
    print(" - models/feature_list.json")

    print("\n📊 Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics

if __name__ == "__main__":
    train_model("data/balanced_ai_human_prompts.csv")
