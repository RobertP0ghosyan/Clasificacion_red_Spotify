import matplotlib

matplotlib.use("Agg")  # Fix Tkinter threading issues

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "dataset"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature columns - THIS MUST MATCH EXACTLY IN BOTH SCRIPTS
FEATURES = [
    "pkt_mean_size",
    "pkt_max_size",
    "pkt_count_up",
    "pkt_count_down",
    "burst_mean",
    "burst_max",
    "bytes_ratio",
    "iat_std",
    "tls_record_mean",
]


# Load dataset
def load_data():
    files = [
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("spotify_classification_") and f.endswith(".csv")
    ]

    if not files:
        raise FileNotFoundError("No dataset found")

    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(DATASET_DIR, f))
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {f}")

    return pd.concat(dfs, ignore_index=True)


# Learning Curve Plot
def plot_learning_curve(model, X, y, title, filename):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="f1_weighted",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
    )

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
    plt.xlabel("Training Samples")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def lasso_feature_selection(X, y, alpha=0.01):
    """Select features using Lasso regression"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=5000, random_state=42)
    lasso.fit(X_scaled, pd.factorize(y)[0])

    selected_features = X.columns[lasso.coef_ != 0].tolist()

    print("\n[Lasso Feature Selection]")
    print(f"  Alpha: {alpha}")
    print(f"  Selected features: {selected_features}")
    print(f"  Features dropped: {set(X.columns) - set(selected_features)}")

    return selected_features


# CONTENT TYPE MODEL (RandomForest)
def train_content_model(df):
    print("\n" + "=" * 60)
    print("=== Training Content Type Model (RandomForest) ===")
    print("=" * 60)

    X = df[FEATURES]
    y = df["content_type"]

    print(f"\nDataset stats:")
    print(f"  Total samples: {len(df)}")
    print(f"  Class distribution:\n{y.value_counts()}")

    # Feature selection
    selected_features = lasso_feature_selection(X, y, alpha=0.01)

    # Fallback safety - ensure minimum features
    if len(selected_features) < 3:
        print("⚠️  Too few features selected, using all features")
        selected_features = FEATURES

    X = df[selected_features]

    # Save selected features for prediction script
    joblib.dump(
        selected_features,
        os.path.join(MODEL_DIR, "content_lasso_features.pkl")
    )
    print(f"\n✓ Saved selected features to content_lasso_features.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    print(f"\nTrain/Test split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Train model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining RandomForest...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"Content Type Model Results:")
    print(f"{'=' * 60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"OOB Score: {model.oob_score_:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, "content_type_rf.pkl"))
    print(f"\n✓ Model saved to content_type_rf.pkl")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title("Content Type Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "content_type_cm.png"))
    plt.close()

    # Learning Curve
    plot_learning_curve(
        model,
        X_train,
        y_train,
        "Learning Curve – Content Type (RF)",
        "lc_content_rf.png"
    )


# GENRE MODEL (XGBoost)
def train_genre_model(df):
    print("\n" + "=" * 60)
    print("=== Training Genre Model (XGBoost) ===")
    print("=" * 60)

    music_df = df[df["content_type"] == "music"].copy()
    music_df = music_df[music_df["genre"] != "unknown"]

    X = music_df[FEATURES]
    y = music_df["genre"]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    joblib.dump(
        label_encoder,
        os.path.join(MODEL_DIR, "genre_label_encoder.pkl")
    )

    # 🔥 LASSO FOR GENRE FEATURES (NEW)
    genre_features = lasso_feature_selection(X, y, alpha=0.01)

    if len(genre_features) < 3:
        print("⚠️ Too few genre features selected, using all features")
        genre_features = FEATURES

    X = music_df[genre_features]

    # Save genre features
    joblib.dump(
        genre_features,
        os.path.join(MODEL_DIR, "genre_lasso_features.pkl")
    )
    print("✓ Saved genre_lasso_features.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        stratify=y_encoded,
        test_size=0.2,
        random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    print("\nGenre Classification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, "genre_xgboost.pkl")
    )
    print("✓ Model saved to genre_xgboost.pkl")



# MAIN
def main():
    print("\n" + "=" * 60)
    print("    SPOTIFY TRAFFIC CLASSIFICATION TRAINING")
    print("=" * 60)

    df = load_data()

    # Remove duplicates if content_id exists
    if "content_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates("content_id")
        print(f"\nRemoved {before - len(df)} duplicate content_ids")

    # Verify required columns
    required_cols = FEATURES + ["content_type"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"\n✓ Dataset ready: {len(df)} samples with {len(FEATURES)} features")

    # Train models
    train_content_model(df)
    train_genre_model(df)

    print("\n" + "=" * 60)
    print("=== TRAINING COMPLETE ===")
    print("=" * 60)
    print(f"Models saved to: {MODEL_DIR}/")
    print(f"  - content_type_rf.pkl")
    print(f"  - genre_xgboost.pkl")
    print(f"  - genre_label_encoder.pkl")
    print(f"  - content_lasso_features.pkl")
    print(f"\nPlots saved to: {RESULTS_DIR}/")
    print(f"  - content_type_cm.png")
    print(f"  - genre_cm.png")
    print(f"  - lc_content_rf.png")
    print(f"  - lc_genre_xgb.png")


if __name__ == "__main__":
    main()