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
    learning_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from xgboost import XGBClassifier

# -----------------------------
# Configuration
# -----------------------------
DATASET_FILE = "dataset/spotify_features.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Updated feature columns based on new extraction script
FEATURES = [
    # Basic packet statistics
    "pkt_count",
    "pkt_avg_len",
    "pkt_max_len",
    "pkt_std_len",
    "pkt_rate",
    "burst_rate",

    # Peak detection features
    "peak_count",
    "peak_mean_height",
    "peak_max_height",
    "peak_frequency",
    "peak_to_avg_ratio",

    # Traffic distribution
    "traffic_cv",
    "traffic_std",
    "traffic_skewness",
    "active_ratio",

    # Initial burst characteristics
    "initial_burst_ratio",
    "initial_burst_max",

    # Inter-arrival times
    "iat_mean",
    "iat_std",
    "iat_median",
]


# -----------------------------
# Load dataset
# -----------------------------
def load_data():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    df = pd.read_csv(DATASET_FILE)
    print(f"Loaded {len(df)} rows from {DATASET_FILE}")

    # Check for missing features
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        print(f"⚠️ Warning: Missing features: {missing_features}")
        # Use only available features
        available_features = [f for f in FEATURES if f in df.columns]
        return df, available_features

    return df, FEATURES


# -----------------------------
# Correlation Matrix
# -----------------------------
def plot_correlation_matrix(df, features, title, filename):
    corr = df[features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()

    print(f"✓ Saved correlation heatmap: {filename}")


# -----------------------------
# Feature Importance Plot
# -----------------------------
def plot_feature_importance(model, feature_names, title, filename):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()

    print(f"✓ Saved feature importance: {filename}")

    # Print top features
    print(f"\nTop 10 Most Important Features for {title}:")
    for i in range(min(10, len(indices))):
        print(f"  {i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")


# -----------------------------
# Learning Curve
# -----------------------------
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
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train", marker='o')
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation", marker='s')
    plt.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.1)
    plt.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1),
                     alpha=0.1)
    plt.xlabel("Training Samples")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()

    print(f"✓ Saved learning curve: {filename}")


# -----------------------------
# Lasso Feature Selection
# -----------------------------
def lasso_feature_selection(X, y, alpha=0.01):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=5000, random_state=42)
    lasso.fit(X_scaled, pd.factorize(y)[0])

    # Get non-zero coefficients
    coefficients = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lasso.coef_
    })

    selected = X.columns[lasso.coef_ != 0].tolist()

    print("\n[Lasso Feature Selection]")
    print(f"  Alpha: {alpha}")
    print(f"  Selected {len(selected)} features: {selected}")
    print(f"  Dropped {len(X.columns) - len(selected)} features")

    # Show top coefficients
    top_features = coefficients.reindex(coefficients['coefficient'].abs().sort_values(ascending=False).index).head(10)
    print("\n  Top 10 Features by Coefficient:")
    for idx, row in top_features.iterrows():
        print(f"    {row['feature']}: {row['coefficient']:.4f}")

    return selected


# -----------------------------
# Content Type Model (RF)
# -----------------------------
def train_content_model(df, features):
    print("\n" + "=" * 60)
    print("=== Training Content Type Model (RandomForest) ===")
    print("=" * 60)

    X = df[features]
    y = df["content_type"]

    print(f"\nDataset: {len(df)} samples")
    print(f"Class distribution:\n{y.value_counts()}")

    selected_features = lasso_feature_selection(X, y)

    if len(selected_features) < 3:
        print("⚠️ Too few features selected, using all features")
        selected_features = features

    joblib.dump(
        selected_features,
        os.path.join(MODEL_DIR, "content_lasso_features.pkl")
    )

    X = df[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth = 6 , # Shallower trees (less complex)
        min_samples_split = 15,  # More conservative splits
        min_samples_leaf = 8,  # Larger leaf nodes
        min_impurity_decrease = 0.001 , # Require minimum improvement
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n[Model Performance]")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"OOB Score: {model.oob_score_:.4f}")
    print("\n", classification_report(y_test, y_pred))

    joblib.dump(model, os.path.join(MODEL_DIR, "content_type_rf.pkl"))
    print(f"✓ Saved model: content_type_rf.pkl")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )
    plt.title("Content Type Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "content_type_cm.png"), dpi=300)
    plt.close()
    print(f"✓ Saved confusion matrix: content_type_cm.png")

    # Feature importance
    plot_feature_importance(
        model,
        selected_features,
        "Content Type - Feature Importance (RF)",
        "content_feature_importance.png"
    )

    # Learning curve
    plot_learning_curve(
        model,
        X_train,
        y_train,
        "Learning Curve – Content Type (RF)",
        "lc_content_rf.png"
    )


# -----------------------------
# Genre Model (XGBoost)
# -----------------------------
def train_genre_model(df, features):
    print("\n" + "=" * 60)
    print("=== Training Genre Model (XGBoost) ===")
    print("=" * 60)

    music_df = df[(df["content_type"] == "music") & (df["genre"] != "unknown")]

    if len(music_df) == 0:
        print("⚠️ No music data available for genre classification")
        return

    X = music_df[features]
    y = music_df["genre"]

    print(f"\nDataset: {len(music_df)} music samples")
    print(f"Genre distribution:\n{y.value_counts()}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    joblib.dump(le, os.path.join(MODEL_DIR, "genre_label_encoder.pkl"))

    genre_features = lasso_feature_selection(X, y)

    if len(genre_features) < 3:
        genre_features = features

    joblib.dump(
        genre_features,
        os.path.join(MODEL_DIR, "genre_lasso_features.pkl")
    )

    X = music_df[genre_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, stratify=y_enc, test_size=0.2, random_state=42
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

    print("\n[Model Performance]")
    print(
        classification_report(
            le.inverse_transform(y_test),
            le.inverse_transform(y_pred)
        )
    )

    joblib.dump(model, os.path.join(MODEL_DIR, "genre_xgboost.pkl"))
    print(f"✓ Saved model: genre_xgboost.pkl")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.title("Genre Classification Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "genre_cm.png"), dpi=300)
    plt.close()
    print(f"✓ Saved confusion matrix: genre_cm.png")

    # Feature importance
    plot_feature_importance(
        model,
        genre_features,
        "Genre Classification - Feature Importance (XGBoost)",
        "genre_feature_importance.png"
    )


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("\n" + "=" * 60)
    print("    SPOTIFY TRAFFIC CLASSIFICATION TRAINING")
    print("    With Peak Detection & Traffic Pattern Features")
    print("=" * 60)

    df, features = load_data()

    print(f"\nUsing {len(features)} features:")
    for f in features:
        print(f"  - {f}")

    if "content_id" in df.columns:
        df = df.drop_duplicates("content_id")
        print(f"\nRemoved duplicates, {len(df)} unique samples remaining")

    # Handle missing values
    print("\nChecking for missing values...")
    missing = df[features].isnull().sum()
    if missing.any():
        print("Missing values found:")
        print(missing[missing > 0])
        df = df.dropna(subset=features)
        print(f"Dropped rows with missing values, {len(df)} samples remaining")
    else:
        print("No missing values found")

    plot_correlation_matrix(
        df,
        features,
        "Feature Correlation Matrix (All Traffic)",
        "feature_correlation_heatmap.png"
    )

    if len(df[df["content_type"] == "music"]) > 0:
        plot_correlation_matrix(
            df[df["content_type"] == "music"],
            features,
            "Feature Correlation Matrix (Music Only)",
            "feature_correlation_music.png"
        )

    train_content_model(df, features)
    train_genre_model(df, features)

    print("\n" + "=" * 60)
    print("=== TRAINING COMPLETE ===")
    print("=" * 60)
    print(f"\nModels saved in: {MODEL_DIR}/")
    print(f"Results saved in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()