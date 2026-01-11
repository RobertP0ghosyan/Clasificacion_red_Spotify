import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATASET_DIR = 'dataset_classification'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

# Create output directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_prepare_data():
    """Load all CSV files from classification dataset directory and combine them"""
    print("=" * 60)
    print("Loading Classification Dataset")
    print("=" * 60)

    # Find all dataset CSV files
    csv_files = [f for f in os.listdir(DATASET_DIR) if
                 f.startswith('spotify_classification_') and f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"No dataset files found in {DATASET_DIR}/")

    print(f"Found {len(csv_files)} dataset file(s):")
    for f in csv_files:
        print(f"  - {f}")

    # Load and combine all datasets
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(DATASET_DIR, csv_file))
        dfs.append(df)
        print(f"  Loaded {len(df)} rows from {csv_file}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


def aggregate_flow_features(df):
    """
    Validate dataset structure and check for duplicates
    """
    print("\n" + "=" * 60)
    print("Validating Dataset Structure")
    print("=" * 60)

    print(f"Total rows (flows) in dataset: {len(df)}")

    # Check for duplicates using content_id
    duplicates = df['content_id'].duplicated().sum()
    if duplicates > 0:
        print(f"\n⚠️  Found {duplicates} duplicate content_id entries")
        print("   Keeping first occurrence of each...")
        df = df.drop_duplicates(subset='content_id', keep='first')
        print(f"   Cleaned dataset: {len(df)} samples")

    print(f"\n📊 Content type distribution:")
    print(df['content_type'].value_counts())

    print(f"\n📊 Genre distribution:")
    print(df['genre'].value_counts())

    # Data is already aggregated (one row per flow)
    return df


def get_feature_columns():
    """Define which columns to use as features for ML models"""
    return [
        # Packet count - important for music vs podcast
        'num_packets',
        # Packet size features
        'pkt_size_mean', 'pkt_size_std', 'pkt_size_cv',
        # Inter-arrival time features
        'inter_mean', 'inter_std', 'inter_cv', 'p95_inter',
        # Burst features
        'burst_mean', 'burst_max',
        # Silence features
        'num_silence_gaps', 'silence_ratio',
        # Rate features
        'pkt_rate',
        # Flow statistics
        'flow_duration'
    ]


def train_content_type_model(df, feature_cols):
    """Train model to classify music vs podcast"""
    print("\n" + "=" * 60)
    print("Training Content Type Classifier (Music vs Podcast)")
    print("=" * 60)

    X = df[feature_cols]
    y = df['content_type']

    print(f"Training samples: {len(X)}")
    print(f"Class distribution:\n{y.value_counts()}")

    # Check if we have enough samples
    if len(y.unique()) < 2:
        print("WARNING: Only one class found! Need both music and podcast samples.")
        return None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train
    print("\nTraining Random Forest...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': pipeline.named_steps['clf'].feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 5 Important Features:")
    print(feature_importance.head())

    # Save model
    model_path = os.path.join(MODEL_DIR, 'content_type_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=pipeline.classes_,
                yticklabels=pipeline.classes_)
    plt.title('Content Type - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'content_type_confusion_matrix.png'))
    plt.close()

    return pipeline


# QUALITY MODEL COMMENTED OUT - NOT NEEDED FOR GENRE CLASSIFICATION
# def train_quality_model(df, feature_cols):
#     """Train model to classify audio quality (low/normal/high/very-high)"""
#     print("\n" + "=" * 60)
#     print("Training Audio Quality Classifier")
#     print("=" * 60)
#
#     # Note: Audio quality is tracked in filename but not as a feature
#     # For genre classification, we want to classify regardless of quality
#     # This helps the model generalize better across different streaming qualities
#
#     X = df[feature_cols]
#     y = df['audio_quality']
#
#     print(f"Training samples: {len(X)}")
#     print(f"Class distribution:\n{y.value_counts()}")
#
#     # Check if we have enough samples
#     if len(y.unique()) < 2:
#         print("WARNING: Only one quality level found! Skipping quality model.")
#         return None
#
#     # Split data - handle case where we might not have enough samples for stratification
#     try:
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )
#     except ValueError:
#         print("Not enough samples for stratification, using random split")
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )
#
#     # Build pipeline
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', RandomForestClassifier(
#             n_estimators=300,
#             max_depth=20,
#             min_samples_split=4,
#             min_samples_leaf=2,
#             random_state=42,
#             n_jobs=-1
#         ))
#     ])
#
#     # Train
#     print("\nTraining Random Forest...")
#     pipeline.fit(X_train, y_train)
#
#     # Evaluate
#     y_pred = pipeline.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#
#     print(f"\nTest Accuracy: {accuracy:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))
#
#     # Feature importance
#     feature_importance = pd.DataFrame({
#         'feature': feature_cols,
#         'importance': pipeline.named_steps['clf'].feature_importances_
#     }).sort_values('importance', ascending=False)
#
#     print("\nTop 5 Important Features:")
#     print(feature_importance.head())
#
#     # Save model
#     model_path = os.path.join(MODEL_DIR, 'audio_quality_model.pkl')
#     joblib.dump(pipeline, model_path)
#     print(f"\nModel saved to: {model_path}")
#
#     # Save confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
#                 xticklabels=pipeline.classes_,
#                 yticklabels=pipeline.classes_)
#     plt.title('Audio Quality - Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig(os.path.join(RESULTS_DIR, 'audio_quality_confusion_matrix.png'))
#     plt.close()
#
#     return pipeline


def train_genre_model(df, feature_cols):
    """Train model to classify music genre (only for music content)"""
    print("\n" + "=" * 60)
    print("Training Genre Classifier (Music Only)")
    print("=" * 60)

    # Filter only music content
    music_df = df[df['content_type'] == 'music'].copy()

    if len(music_df) == 0:
        print("WARNING: No music samples found! Skipping genre model.")
        return None

    # Remove 'unknown' genre if present
    music_df = music_df[music_df['genre'] != 'unknown']

    if len(music_df) == 0:
        print("WARNING: No music samples with known genre! Skipping genre model.")
        return None

    X = music_df[feature_cols]
    y = music_df['genre']

    print(f"Training samples: {len(X)}")
    print(f"Genre distribution:\n{y.value_counts()}")

    # Check if we have enough samples
    if len(y.unique()) < 2:
        print("WARNING: Only one genre found! Need multiple genres for classification.")
        return None

    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("Not enough samples for stratification, using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train
    print("\nTraining Random Forest...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': pipeline.named_steps['clf'].feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 5 Important Features:")
    print(feature_importance.head())

    # Save model
    model_path = os.path.join(MODEL_DIR, 'genre_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=pipeline.classes_,
                yticklabels=pipeline.classes_)
    plt.title('Genre - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'genre_confusion_matrix.png'))
    plt.close()

    return pipeline


def save_training_summary(df, models):
    """Save a summary of the training results"""
    summary = {
        'total_flows': len(df),
        'content_types': df['content_type'].value_counts().to_dict(),
        'genres': df['genre'].value_counts().to_dict(),
        'models_trained': list(models.keys()),
        'feature_columns': get_feature_columns()
    }

    # Save as text file
    summary_path = os.path.join(RESULTS_DIR, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SPOTIFY GENRE/CONTENT CLASSIFICATION - TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total Flows: {summary['total_flows']}\n\n")

        f.write("Content Type Distribution:\n")
        for k, v in summary['content_types'].items():
            f.write(f"  {k}: {v}\n")

        f.write("\nGenre Distribution:\n")
        for k, v in summary['genres'].items():
            f.write(f"  {k}: {v}\n")

        f.write(f"\nModels Trained: {', '.join(summary['models_trained'])}\n")

        f.write(f"\nFeatures Used ({len(summary['feature_columns'])}):\n")
        for feat in summary['feature_columns']:
            f.write(f"  - {feat}\n")

        f.write("\nNote: Audio quality model not trained for genre classification.\n")
        f.write("Quality is tracked in filename but not used as a feature to ensure\n")
        f.write("the model generalizes across different streaming qualities.\n")

    print(f"\nTraining summary saved to: {summary_path}")


def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("SPOTIFY GENRE/CONTENT CLASSIFICATION - MODEL TRAINING")
    print("=" * 60 + "\n")

    # Load and prepare data
    df_raw = load_and_prepare_data()
    df = aggregate_flow_features(df_raw)

    # Get feature columns
    feature_cols = get_feature_columns()
    print(f"\nUsing {len(feature_cols)} features for training")

    # Train models
    models = {}

    # 1. Content Type Model (Music vs Podcast)
    content_model = train_content_type_model(df, feature_cols)
    if content_model:
        models['content_type'] = content_model

    # 2. Audio Quality Model - COMMENTED OUT
    # Quality is tracked in filename but not used for classification
    # This helps the model generalize better across different streaming qualities
    # quality_model = train_quality_model(df, feature_cols)
    # if quality_model:
    #     models['audio_quality'] = quality_model

    # 3. Genre Model (Music only)
    genre_model = train_genre_model(df, feature_cols)
    if genre_model:
        models['genre'] = genre_model

    # Save training summary
    save_training_summary(df, models)

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Models trained: {len(models)}/2 (Content Type + Genre)")
    print(f"Models saved in: {MODEL_DIR}/")
    print(f"Results saved in: {RESULTS_DIR}/")
    print("\nTrained models:")
    for model_name in models.keys():
        print(f"  ✓ {model_name}_model.pkl")
    print("\n🎯 Purpose:")
    print("  • Content Type: Classify Music vs Podcast")
    print("  • Genre: Classify specific music genres")
    print("\n⚠️  Note: Audio quality model NOT trained")
    print("  Quality is tracked but not used as a feature to ensure")
    print("  the model works across all streaming qualities.")
    print("\nYou can now use these models for real-time traffic classification!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()