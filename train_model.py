import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json

# Configuración
DATASET_DIR = 'dataset'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

class SpotifyQualityClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_datasets(self):
        """Cargar todos los CSV del directorio dataset"""
        print("=" * 70)
        print("CARGANDO DATASETS")
        print("=" * 70)
        
        csv_files = glob.glob(f"{DATASET_DIR}/spotify_classification_*.csv")
        
        if not csv_files:
            raise Exception(f"No se encontraron archivos CSV en {DATASET_DIR}/")
        
        print(f"\nArchivos encontrados: {len(csv_files)}")
        
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            print(f"  ✓ {os.path.basename(csv_file)}: {len(df)} muestras")
            dfs.append(df)
        
        # Combinar todos los datasets
        combined_df = pd.concat(dfs, ignore_index=True)
        
        print(f"\n📊 Dataset combinado: {len(combined_df)} muestras totales")
        print("\nDistribución por calidad:")
        print(combined_df['quality'].value_counts().sort_index())
        
        return combined_df
    
    def prepare_data(self, df):
        """Preparar datos para entrenamiento"""
        print("\n" + "=" * 70)
        print("PREPARANDO DATOS")
        print("=" * 70)
        
        # Eliminar columnas no útiles para el modelo
        columns_to_drop = ['content_type', 'genre', 'content_id']
        
        # Separar features y target
        X = df.drop(columns_to_drop + ['quality'], axis=1)
        y = df['quality']
        
        # Guardar nombres de las columnas de features
        self.feature_columns = X.columns.tolist()
        print(f"\n✓ Features seleccionadas: {len(self.feature_columns)}")
        print(f"  Ejemplos: {self.feature_columns[:5]}")
        
        # Manejar valores infinitos o NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Codificar las etiquetas de calidad
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n✓ Clases de calidad detectadas: {list(self.label_encoder.classes_)}")
        print(f"  Mapeo: {dict(enumerate(self.label_encoder.classes_))}")
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\n✓ Split realizado:")
        print(f"  Train: {len(X_train)} muestras")
        print(f"  Test:  {len(X_test)} muestras")
        
        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n✓ Features normalizadas con StandardScaler")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_model(self, input_dim, num_classes):
        """Construir red neuronal"""
        print("\n" + "=" * 70)
        print("CONSTRUYENDO RED NEURONAL")
        print("=" * 70)
        
        model = keras.Sequential([
            # Capa de entrada
            layers.Input(shape=(input_dim,)),
            
            # Primera capa densa con dropout
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Segunda capa densa
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Tercera capa densa
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Cuarta capa densa
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Capa de salida
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n✓ Arquitectura del modelo:")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test):
        """Entrenar el modelo"""
        print("\n" + "=" * 70)
        print("ENTRENANDO MODELO")
        print("=" * 70)
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Entrenar
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluar el modelo"""
        print("\n" + "=" * 70)
        print("EVALUACIÓN DEL MODELO")
        print("=" * 70)
        
        # Predicciones
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✓ Accuracy: {accuracy * 100:.2f}%")
        
        # Classification Report
        print("\n📊 Reporte de Clasificación:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion Matrix
        print("\n📊 Matriz de Confusión:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Mostrar de forma más legible
        print("\nMatriz de Confusión (formato tabla):")
        print(f"{'':12}", end='')
        for label in self.label_encoder.classes_:
            print(f"{label:12}", end='')
        print()
        
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"{label:12}", end='')
            for j in range(len(self.label_encoder.classes_)):
                print(f"{cm[i][j]:12}", end='')
            print()
        
        return accuracy
    
    def save_model(self):
        """Guardar modelo y componentes"""
        print("\n" + "=" * 70)
        print("GUARDANDO MODELO")
        print("=" * 70)
        
        # Guardar modelo de Keras
        model_path = f"{MODEL_DIR}/spotify_quality_model.h5"
        self.model.save(model_path)
        print(f"✓ Modelo guardado: {model_path}")
        
        # Guardar scaler
        scaler_path = f"{MODEL_DIR}/scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler guardado: {scaler_path}")
        
        # Guardar label encoder
        encoder_path = f"{MODEL_DIR}/label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        print(f"✓ Label encoder guardado: {encoder_path}")
        
        # Guardar nombres de features
        features_path = f"{MODEL_DIR}/feature_columns.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f)
        print(f"✓ Feature columns guardadas: {features_path}")
        
        # Guardar metadata
        metadata = {
            'num_features': len(self.feature_columns),
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_)
        }
        metadata_path = f"{MODEL_DIR}/model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata guardada: {metadata_path}")
        
        print("\n" + "=" * 70)
        print("✅ TODOS LOS ARCHIVOS GUARDADOS EXITOSAMENTE")
        print("=" * 70)
    
    def run_training_pipeline(self):
        """Pipeline completo de entrenamiento"""
        try:
            # 1. Cargar datos
            df = self.load_datasets()
            
            # 2. Preparar datos
            X_train, X_test, y_train, y_test = self.prepare_data(df)
            
            # 3. Construir modelo
            self.model = self.build_model(
                input_dim=X_train.shape[1],
                num_classes=len(self.label_encoder.classes_)
            )
            
            # 4. Entrenar
            history = self.train(X_train, y_train, X_test, y_test)
            
            # 5. Evaluar
            accuracy = self.evaluate(X_test, y_test)
            
            # 6. Guardar
            self.save_model()
            
            print(f"\n{'=' * 70}")
            print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            print(f"{'=' * 70}")
            print(f"Accuracy final: {accuracy * 100:.2f}%")
            print(f"Modelo guardado en: {MODEL_DIR}/")
            print(f"{'=' * 70}\n")
            
        except Exception as e:
            print(f"\n❌ ERROR durante el entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    print("=" * 70)
    print("SPOTIFY QUALITY CLASSIFIER - ENTRENAMIENTO")
    print("=" * 70)
    print("\n📋 Este script:")
    print("  1. Carga todos los CSV de la carpeta 'dataset/'")
    print("  2. Prepara y normaliza los datos")
    print("  3. Construye una red neuronal")
    print("  4. Entrena el modelo")
    print("  5. Evalúa el rendimiento")
    print("  6. Guarda el modelo en 'models/'")
    print("\n⚠️  Requisitos:")
    print("  - pip install tensorflow scikit-learn pandas numpy joblib")
    print("  - Tener archivos CSV en la carpeta 'dataset/'")
    print("=" * 70 + "\n")
    
    # Verificar que existe el directorio de datasets
    if not os.path.exists(DATASET_DIR):
        print(f"❌ ERROR: No se encuentra el directorio '{DATASET_DIR}/'")
        print("   Asegúrate de haber ejecutado el script de captura primero.")
        exit(1)
    
    csv_files = glob.glob(f"{DATASET_DIR}/spotify_classification_*.csv")
    if not csv_files:
        print(f"❌ ERROR: No hay archivos CSV en '{DATASET_DIR}/'")
        print("   Ejecuta el script de captura primero para generar datos.")
        exit(1)
    
    input("Presiona ENTER para comenzar el entrenamiento...")
    
    # Crear y ejecutar clasificador
    classifier = SpotifyQualityClassifier()
    classifier.run_training_pipeline()
