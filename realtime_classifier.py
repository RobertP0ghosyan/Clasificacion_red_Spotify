import numpy as np
import time
import json
import os
from scapy.all import sniff, IP, TCP
from collections import deque
import tensorflow as tf
from tensorflow import keras
import joblib
from datetime import datetime

# Configuración
MODEL_DIR = 'models'
CAPTURE_DURATION = 60  # Capturar durante 60 segundos antes de clasificar
INTERFACE = "enp0s3"  # Cambiar según tu interfaz

class RealtimeSpotifyClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.current_capture = []
        self.last_packet_time = None
        
        # Cargar modelo y componentes
        self.load_model()
    
    def load_model(self):
        """Cargar modelo entrenado y componentes"""
        print("=" * 70)
        print("CARGANDO MODELO")
        print("=" * 70)
        
        # Verificar que existen los archivos
        required_files = [
            f"{MODEL_DIR}/spotify_quality_model.h5",
            f"{MODEL_DIR}/scaler.pkl",
            f"{MODEL_DIR}/label_encoder.pkl",
            f"{MODEL_DIR}/feature_columns.json"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                raise Exception(f"Archivo no encontrado: {file}\nEjecuta primero train_model.py")
        
        # Cargar modelo
        self.model = keras.models.load_model(f"{MODEL_DIR}/spotify_quality_model.h5")
        print(f"✓ Modelo cargado")
        
        # Cargar scaler
        self.scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        print(f"✓ Scaler cargado")
        
        # Cargar label encoder
        self.label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
        print(f"✓ Label encoder cargado")
        print(f"  Clases: {list(self.label_encoder.classes_)}")
        
        # Cargar nombres de features
        with open(f"{MODEL_DIR}/feature_columns.json", 'r') as f:
            self.feature_columns = json.load(f)
        print(f"✓ Feature columns cargadas: {len(self.feature_columns)} features")
        
        print("=" * 70 + "\n")
    
    def packet_callback(self, packet):
        """Callback para cada paquete capturado"""
        arrival_time = packet.time
        payload_size = len(packet)
        
        # TCP/IP specific features
        tcp_flags = 0
        ip_ttl = 0
        tcp_window = 0
        ip_len = 0
        
        if IP in packet:
            ip_ttl = packet[IP].ttl
            ip_len = packet[IP].len
            
        if TCP in packet:
            tcp_flags = packet[TCP].flags
            tcp_window = packet[TCP].window

        inter_arrival = 0
        if self.last_packet_time is not None:
            inter_arrival = arrival_time - self.last_packet_time
        self.last_packet_time = arrival_time

        self.current_capture.append({
            "arrival_time": arrival_time,
            "payload_size": payload_size,
            "inter_arrival": inter_arrival,
            "tcp_flags": int(tcp_flags),
            "ip_ttl": ip_ttl,
            "tcp_window": tcp_window,
            "ip_len": ip_len
        })
    
    def compute_flow_features(self):
        """Calcular las mismas features que durante el entrenamiento"""
        if not self.current_capture:
            print("⚠️  Warning: No se capturaron paquetes!")
            return self._empty_features()

        # Extract arrays
        pkt_sizes = np.array([p["payload_size"] for p in self.current_capture])
        inter_arrivals = np.array([p["inter_arrival"] for p in self.current_capture[1:]])
        timestamps = np.array([p["arrival_time"] for p in self.current_capture])
        tcp_flags = np.array([p["tcp_flags"] for p in self.current_capture])
        ip_ttls = np.array([p["ip_ttl"] for p in self.current_capture])
        tcp_windows = np.array([p["tcp_window"] for p in self.current_capture])
        ip_lens = np.array([p["ip_len"] for p in self.current_capture])

        features = {}
        
        # ==================== PACKET SIZE FEATURES ====================
        features["pkt_size_mean"] = np.mean(pkt_sizes)
        features["pkt_size_std"] = np.std(pkt_sizes)
        features["pkt_size_min"] = np.min(pkt_sizes)
        features["pkt_size_max"] = np.max(pkt_sizes)
        features["pkt_size_median"] = np.median(pkt_sizes)
        features["pkt_size_cv"] = features["pkt_size_std"] / features["pkt_size_mean"] if features["pkt_size_mean"] else 0
        features["pkt_size_p25"] = np.percentile(pkt_sizes, 25)
        features["pkt_size_p75"] = np.percentile(pkt_sizes, 75)
        features["pkt_size_p95"] = np.percentile(pkt_sizes, 95)
        
        # Packet size distribution (histogram bins)
        hist, _ = np.histogram(pkt_sizes, bins=5, range=(0, 2000))
        for i, count in enumerate(hist):
            features[f"pkt_size_bin_{i}"] = count

        # ==================== INTER-ARRIVAL TIME FEATURES ====================
        if len(inter_arrivals) > 0:
            features["inter_mean"] = np.mean(inter_arrivals)
            features["inter_std"] = np.std(inter_arrivals)
            features["inter_min"] = np.min(inter_arrivals)
            features["inter_max"] = np.max(inter_arrivals)
            features["inter_median"] = np.median(inter_arrivals)
            features["inter_cv"] = features["inter_std"] / features["inter_mean"] if features["inter_mean"] else 0
            features["inter_p25"] = np.percentile(inter_arrivals, 25)
            features["inter_p75"] = np.percentile(inter_arrivals, 75)
            features["inter_p95"] = np.percentile(inter_arrivals, 95)
        else:
            features["inter_mean"] = 0
            features["inter_std"] = 0
            features["inter_min"] = 0
            features["inter_max"] = 0
            features["inter_median"] = 0
            features["inter_cv"] = 0
            features["inter_p25"] = 0
            features["inter_p75"] = 0
            features["inter_p95"] = 0

        # ==================== BURST FEATURES ====================
        BURST_WINDOW = 0.5
        bursts = []
        window = []
        for t in timestamps:
            window = [x for x in window if t - x <= BURST_WINDOW]
            window.append(t)
            bursts.append(len(window))
        
        features["burst_mean"] = np.mean(bursts) if bursts else 0
        features["burst_max"] = max(bursts) if bursts else 0
        features["burst_std"] = np.std(bursts) if bursts else 0
        features["burst_median"] = np.median(bursts) if bursts else 0

        # ==================== SILENCE/GAP FEATURES ====================
        SILENCE_THRESHOLD = 2.0
        silence_gaps = [x for x in inter_arrivals if x > SILENCE_THRESHOLD]
        features["num_silence_gaps"] = len(silence_gaps)
        features["silence_ratio"] = sum(silence_gaps) / sum(inter_arrivals) if sum(inter_arrivals) > 0 else 0
        features["avg_silence_duration"] = np.mean(silence_gaps) if silence_gaps else 0

        # ==================== FLOW DURATION & RATE FEATURES ====================
        flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        features["flow_duration"] = flow_duration
        features["pkt_rate"] = len(pkt_sizes) / flow_duration if flow_duration > 0 else 0
        features["byte_rate"] = np.sum(pkt_sizes) / flow_duration if flow_duration > 0 else 0
        
        # ==================== DIRECTIONALITY FEATURES ====================
        small_pkts = np.sum(pkt_sizes < 100)
        large_pkts = np.sum(pkt_sizes >= 100)
        features["small_pkt_ratio"] = small_pkts / len(pkt_sizes) if len(pkt_sizes) > 0 else 0
        features["large_pkt_ratio"] = large_pkts / len(pkt_sizes) if len(pkt_sizes) > 0 else 0

        # ==================== TCP FLAGS FEATURES ====================
        features["tcp_syn_count"] = np.sum((tcp_flags & 0x02) != 0)
        features["tcp_ack_count"] = np.sum((tcp_flags & 0x10) != 0)
        features["tcp_psh_count"] = np.sum((tcp_flags & 0x08) != 0)
        features["tcp_fin_count"] = np.sum((tcp_flags & 0x01) != 0)
        features["tcp_rst_count"] = np.sum((tcp_flags & 0x04) != 0)

        # ==================== IP/TCP HEADER FEATURES ====================
        features["ip_ttl_mean"] = np.mean(ip_ttls) if len(ip_ttls) > 0 else 0
        features["ip_ttl_std"] = np.std(ip_ttls) if len(ip_ttls) > 0 else 0
        features["tcp_window_mean"] = np.mean(tcp_windows) if len(tcp_windows) > 0 else 0
        features["tcp_window_std"] = np.std(tcp_windows) if len(tcp_windows) > 0 else 0
        features["ip_len_mean"] = np.mean(ip_lens) if len(ip_lens) > 0 else 0

        # ==================== TEMPORAL PATTERNS ====================
        quarter_size = len(pkt_sizes) // 4
        if quarter_size > 0:
            for i in range(4):
                start = i * quarter_size
                end = start + quarter_size if i < 3 else len(pkt_sizes)
                quarter_pkts = pkt_sizes[start:end]
                features[f"quarter_{i}_mean_size"] = np.mean(quarter_pkts)
                features[f"quarter_{i}_pkt_count"] = len(quarter_pkts)
        else:
            for i in range(4):
                features[f"quarter_{i}_mean_size"] = 0
                features[f"quarter_{i}_pkt_count"] = 0

        # ==================== ADDITIONAL STATISTICAL FEATURES ====================
        features["num_packets"] = len(pkt_sizes)
        features["total_bytes"] = np.sum(pkt_sizes)
        
        from scipy import stats
        features["pkt_size_skewness"] = stats.skew(pkt_sizes) if len(pkt_sizes) > 1 else 0
        features["pkt_size_kurtosis"] = stats.kurtosis(pkt_sizes) if len(pkt_sizes) > 1 else 0
        
        if len(inter_arrivals) > 1:
            features["inter_skewness"] = stats.skew(inter_arrivals)
            features["inter_kurtosis"] = stats.kurtosis(inter_arrivals)
        else:
            features["inter_skewness"] = 0
            features["inter_kurtosis"] = 0

        return features
    
    def _empty_features(self):
        """Return empty feature dict when no packets captured"""
        empty = {
            "pkt_size_mean": 0, "pkt_size_std": 0, "pkt_size_min": 0, "pkt_size_max": 0,
            "pkt_size_median": 0, "pkt_size_cv": 0, "pkt_size_p25": 0, "pkt_size_p75": 0, "pkt_size_p95": 0,
            "inter_mean": 0, "inter_std": 0, "inter_min": 0, "inter_max": 0, "inter_median": 0,
            "inter_cv": 0, "inter_p25": 0, "inter_p75": 0, "inter_p95": 0,
            "burst_mean": 0, "burst_max": 0, "burst_std": 0, "burst_median": 0,
            "num_silence_gaps": 0, "silence_ratio": 0, "avg_silence_duration": 0,
            "flow_duration": 0, "pkt_rate": 0, "byte_rate": 0,
            "small_pkt_ratio": 0, "large_pkt_ratio": 0,
            "tcp_syn_count": 0, "tcp_ack_count": 0, "tcp_psh_count": 0, "tcp_fin_count": 0, "tcp_rst_count": 0,
            "ip_ttl_mean": 0, "ip_ttl_std": 0, "tcp_window_mean": 0, "tcp_window_std": 0, "ip_len_mean": 0,
            "num_packets": 0, "total_bytes": 0,
            "pkt_size_skewness": 0, "pkt_size_kurtosis": 0, "inter_skewness": 0, "inter_kurtosis": 0
        }
        for i in range(5):
            empty[f"pkt_size_bin_{i}"] = 0
        for i in range(4):
            empty[f"quarter_{i}_mean_size"] = 0
            empty[f"quarter_{i}_pkt_count"] = 0
        return empty
    
    def classify_traffic(self):
        """Capturar tráfico y clasificar"""
        print("\n" + "=" * 70)
        print(f"CAPTURANDO TRÁFICO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Resetear captura
        self.current_capture = []
        self.last_packet_time = None
        
        # Capturar paquetes
        print(f"🎧 Escuchando en interfaz '{INTERFACE}'...")
        print(f"⏱️  Capturando durante {CAPTURE_DURATION} segundos...")
        print("   (Asegúrate de que Spotify esté reproduciendo)")
        
        packets = sniff(
            iface=INTERFACE,
            filter="tcp port 443",
            prn=self.packet_callback,
            timeout=CAPTURE_DURATION,
            store=False  # No almacenar para ahorrar memoria
        )
        
        print(f"\n✓ Captura finalizada: {len(self.current_capture)} paquetes")
        
        if len(self.current_capture) < 10:
            print("⚠️  Muy pocos paquetes capturados. ¿Está Spotify reproduciendo?")
            return None
        
        # Calcular features
        print("🔬 Extrayendo características del tráfico...")
        features = self.compute_flow_features()
        
        # Convertir a array ordenado según feature_columns
        feature_array = np.array([features[col] for col in self.feature_columns])
        
        # Manejar valores infinitos o NaN
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalizar
        feature_array_scaled = self.scaler.transform(feature_array.reshape(1, -1))
        
        # Predecir
        print("🤖 Clasificando con la red neuronal...")
        predictions = self.model.predict(feature_array_scaled, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_quality = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = predictions[0][predicted_class_idx] * 100
        
        # Mostrar resultados
        print("\n" + "=" * 70)
        print("📊 RESULTADO DE LA CLASIFICACIÓN")
        print("=" * 70)
        print(f"\n🎵 Calidad detectada: {predicted_quality.upper()}")
        print(f"📈 Confianza: {confidence:.2f}%")
        
        print("\n🎯 Probabilidades por clase:")
        for i, quality_class in enumerate(self.label_encoder.classes_):
            prob = predictions[0][i] * 100
            bar = "█" * int(prob / 2)
            print(f"  {quality_class:10} {prob:6.2f}% {bar}")
        
        print("\n📦 Estadísticas del tráfico capturado:")
        print(f"  Paquetes:      {features['num_packets']}")
        print(f"  Total bytes:   {features['total_bytes']:,.0f}")
        print(f"  Duración:      {features['flow_duration']:.2f}s")
        print(f"  Tasa pkts/s:   {features['pkt_rate']:.2f}")
        print(f"  Tasa bytes/s:  {features['byte_rate']:,.0f}")
        print(f"  Tamaño medio:  {features['pkt_size_mean']:.2f} bytes")
        
        print("=" * 70 + "\n")
        
        return predicted_quality, confidence
    
    def run_continuous_monitoring(self):
        """Monitoreo continuo del tráfico"""
        print("=" * 70)
        print("🚀 MODO MONITOREO CONTINUO ACTIVADO")
        print("=" * 70)
        print("\nEl sistema capturará tráfico cada 60 segundos y clasificará la calidad.")
        print("Presiona Ctrl+C para detener.\n")
        
        capture_count = 0
        
        try:
            while True:
                capture_count += 1
                print(f"\n{'#' * 70}")
                print(f"CAPTURA #{capture_count}")
                print(f"{'#' * 70}")
                
                result = self.classify_traffic()
                
                if result:
                    quality, confidence = result
                    print(f"✅ Clasificación completada: {quality.upper()} ({confidence:.2f}%)")
                else:
                    print("❌ No se pudo clasificar (sin tráfico)")
                
                # Esperar antes de la próxima captura
                wait_time = 10
                print(f"\n⏳ Esperando {wait_time} segundos antes de la próxima captura...")
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoreo detenido por el usuario")
            print(f"Total de capturas realizadas: {capture_count}")


if __name__ == "__main__":
    print("=" * 70)
    print("SPOTIFY QUALITY CLASSIFIER - TIEMPO REAL")
    print("=" * 70)
    print("\n📋 Este script:")
    print("  1. Carga el modelo entrenado")
    print("  2. Captura tráfico de Spotify en tiempo real")
    print("  3. Clasifica la calidad: low, normal, high, very_high")
    print("\n⚠️  Requisitos:")
    print("  - Ejecutar con sudo (captura de red)")
    print("  - Tener el modelo entrenado en 'models/'")
    print("  - Spotify reproduciendo contenido")
    print("  - pip install tensorflow scikit-learn scapy scipy numpy joblib")
    print("=" * 70 + "\n")
    
    # Verificar que se ejecuta con sudo
    if os.geteuid() != 0:
        print("❌ ERROR: Este script requiere privilegios de root")
        print("   Ejecuta con: sudo python3 realtime_classifier.py")
        exit(1)
    
    # Verificar que existe el modelo
    if not os.path.exists(MODEL_DIR):
        print(f"❌ ERROR: No se encuentra el directorio '{MODEL_DIR}/'")
        print("   Ejecuta primero train_model.py para entrenar el modelo.")
        exit(1)
    
    # Preguntar modo de operación
    print("Selecciona el modo de operación:")
    print("  1. Captura única (una sola clasificación)")
    print("  2. Monitoreo continuo (clasificaciones repetidas)")
    
    while True:
        choice = input("\nOpción (1 o 2): ").strip()
        if choice in ['1', '2']:
            break
        print("❌ Opción inválida. Introduce 1 o 2.")
    
    # Crear clasificador
    classifier = RealtimeSpotifyClassifier()
    
    if choice == '1':
        input("\n🎵 Abre Spotify y reproduce contenido, luego presiona ENTER...")
        classifier.classify_traffic()
    else:
        classifier.run_continuous_monitoring()
