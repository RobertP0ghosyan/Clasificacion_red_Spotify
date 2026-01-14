#!/usr/bin/env python3

import joblib
import pandas as pd
import os
import json
import numpy as np
from scapy.all import sniff, wrpcap, IP, TCP
from scipy import stats

# -----------------------------
# Configuration
# -----------------------------
CAPTURE_FILE = "live_capture.pcap"
INTERFACE = "enp0s3"  # Change if needed (e.g., "eth0", "en0")
CAPTURE_DURATION = 60  # seconds
MODEL_DIR = "models"
IP_CACHE_FILE = "spotify_ips.json"

LOCAL_PREFIXES = ("192.168.", "10.", "172.16.", "127.")

# -----------------------------
# Load cached Spotify IPs
# -----------------------------
try:
    with open(IP_CACHE_FILE, "r") as f:
        data = json.load(f)
        SPOTIFY_IPS = set(data.get("ips", []))
        print(f"[*] Loaded {len(SPOTIFY_IPS)} cached Spotify IPs")
except FileNotFoundError:
    SPOTIFY_IPS = set()
    print("[*] No cached IPs found")

# Known Spotify IP prefixes (Google Cloud)
SPOTIFY_IP_PREFIXES = [
    "35.186.", "104.154.", "35.184.",
    "35.185.", "104.199.", "34.120.",
    "34.117.", "35.190.", "35.244."
]


# -----------------------------
# Load Models
# -----------------------------
def load_models():
    """Load all trained models and encoders"""
    print("\n" + "=" * 60)
    print("Loading trained models...")
    print("=" * 60)

    try:
        content_model = joblib.load(f"{MODEL_DIR}/content_type_rf.pkl")
        genre_model = joblib.load(f"{MODEL_DIR}/genre_xgboost.pkl")
        label_encoder = joblib.load(f"{MODEL_DIR}/genre_label_encoder.pkl")
        content_features = joblib.load(f"{MODEL_DIR}/content_lasso_features.pkl")
        genre_features = joblib.load("models/genre_lasso_features.pkl")

        print(f"  ✓ Content type model loaded")
        print(f"  ✓ Genre model loaded")
        print(f"  ✓ Label encoder loaded")
        print(f"  ✓ Feature list loaded: {len(content_features)} features")
        print(f"\nModel expects these features:")
        for feat in content_features:
            print(f"    - {feat}")

        return content_model, genre_model, label_encoder, content_features, genre_features

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nMake sure you've run train.py first!")
        raise


# -----------------------------
# Packet Filtering
# -----------------------------
def is_spotify_packet(packet):
    """Check if packet is from/to Spotify servers"""
    if not packet.haslayer(IP):
        return False

    src, dst = packet[IP].src, packet[IP].dst

    # Check cached IPs first (faster)
    if SPOTIFY_IPS and (src in SPOTIFY_IPS or dst in SPOTIFY_IPS):
        return True

    # Check known prefixes
    return any(
        src.startswith(p) or dst.startswith(p)
        for p in SPOTIFY_IP_PREFIXES
    )


# -----------------------------
# Feature Extraction
# -----------------------------
def extract_tls_records(packet):
    """Extract TLS record lengths from packet payload"""
    tls_records = []

    if not packet.haslayer(TCP):
        return tls_records

    try:
        # Get raw TCP payload
        if hasattr(packet[TCP], 'load'):
            payload = bytes(packet[TCP].load)
        else:
            return tls_records

        offset = 0
        while offset + 5 <= len(payload):
            content_type = payload[offset]

            # TLS content types: 0x14 (ChangeCipherSpec), 0x15 (Alert),
            # 0x16 (Handshake), 0x17 (Application Data), 0x18 (Heartbeat)
            if content_type < 0x14 or content_type > 0x18:
                break

            # Verify TLS version (0x0301 = TLS 1.0, 0x0303 = TLS 1.2, 0x0304 = TLS 1.3)
            version = (payload[offset + 1] << 8) | payload[offset + 2]
            if version < 0x0301 or version > 0x0304:
                break

            # Extract record length
            length = (payload[offset + 3] << 8) | payload[offset + 4]

            # Sanity check: TLS record max size is 16KB
            if length > 16384 or length == 0:
                break

            tls_records.append(length)
            offset += 5 + length

    except Exception:
        pass

    return tls_records


def calculate_bursts(timestamps, gap=0.1):
    """Calculate packet burst sizes"""
    if len(timestamps) <= 1:
        return [1]

    bursts, count = [], 1
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] < gap:
            count += 1
        else:
            bursts.append(count)
            count = 1
    bursts.append(count)
    return bursts


def compute_flow_features(packets):
    """Extract comprehensive traffic features from packet list"""
    if not packets:
        return None, None

    print(f"\n[*] Extracting features from {len(packets)} packets...")

    # Parse all packets
    data = []
    for p in packets:
        ts = float(p.time)
        size = len(p)

        # Determine direction based on source IP
        direction = "upstream" if p[IP].src.startswith(LOCAL_PREFIXES) else "downstream"

        # Extract TLS records
        tls_records = extract_tls_records(p)

        data.append({
            'timestamp': ts,
            'size': size,
            'direction': direction,
            'tls_records': tls_records
        })

    # Separate by direction
    timestamps = sorted([d['timestamp'] for d in data])
    all_sizes = [d['size'] for d in data]
    upstream_sizes = [d['size'] for d in data if d['direction'] == "upstream"]
    downstream_sizes = [d['size'] for d in data if d['direction'] == "downstream"]
    all_tls = [r for d in data for r in d['tls_records']]

    # Calculate inter-arrival times
    iat = np.diff(timestamps) if len(timestamps) > 1 else [0]

    # Calculate bursts
    bursts = calculate_bursts(timestamps)

    # Flow duration and packet rate
    flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    packet_rate = len(packets) / flow_duration if flow_duration > 0 else 0

    # Bytes statistics
    upstream_bytes = sum(upstream_sizes)
    downstream_bytes = sum(downstream_sizes)
    total_bytes = upstream_bytes + downstream_bytes
    bytes_ratio = downstream_bytes / upstream_bytes if upstream_bytes > 0 else 0

    # Packet count ratio
    pkt_count_up = len(upstream_sizes)
    pkt_count_down = len(downstream_sizes)
    pkt_count_ratio = pkt_count_down / pkt_count_up if pkt_count_up > 0 else 0

    # Coefficient of Variation calculations
    def safe_cv(arr):
        """Calculate CV, return 0 if can't calculate"""
        if len(arr) == 0:
            return 0
        mean_val = np.mean(arr)
        if mean_val == 0:
            return 0
        return np.std(arr) / mean_val

    pkt_cv = safe_cv(all_sizes)
    downstream_cv = safe_cv(downstream_sizes)
    burst_cv = safe_cv(bursts)
    iat_cv = safe_cv(iat)
    tls_cv = safe_cv(all_tls)

    # Burst statistics
    burst_count = len(bursts)
    burst_frequency = burst_count / flow_duration if flow_duration > 0 else 0

    # Packet size distribution
    large_pkt_count = sum(1 for size in downstream_sizes if size > 1400)
    large_pkt_ratio = large_pkt_count / len(downstream_sizes) if len(downstream_sizes) > 0 else 0

    small_pkt_count = sum(1 for size in all_sizes if size < 100)
    small_pkt_ratio = small_pkt_count / len(all_sizes) if len(all_sizes) > 0 else 0

    # Downstream throughput
    downstream_throughput = downstream_bytes / flow_duration if flow_duration > 0 else 0

    # Compute all features (matching the extractor)
    features = {
        # Basic packet statistics
        'pkt_mean_size': np.mean(all_sizes) if all_sizes else 0,
        'pkt_max_size': max(all_sizes) if all_sizes else 0,
        'pkt_min_size': min(all_sizes) if all_sizes else 0,
        'pkt_std_size': np.std(all_sizes) if all_sizes else 0,
        'pkt_cv': pkt_cv,

        # Directional packet counts
        'pkt_count_up': pkt_count_up,
        'pkt_count_down': pkt_count_down,
        'pkt_count_ratio': pkt_count_ratio,

        # Directional packet sizes
        'downstream_mean_size': np.mean(downstream_sizes) if downstream_sizes else 0,
        'downstream_cv': downstream_cv,
        'upstream_mean_size': np.mean(upstream_sizes) if upstream_sizes else 0,

        # Burst characteristics
        'burst_mean': np.mean(bursts) if bursts else 1,
        'burst_max': max(bursts) if bursts else 1,
        'burst_count': burst_count,
        'burst_frequency': burst_frequency,
        'burst_cv': burst_cv,

        # Bytes and throughput
        'bytes_ratio': bytes_ratio,
        'total_bytes': total_bytes,
        'downstream_throughput': downstream_throughput,

        # Inter-arrival times
        'iat_mean': np.mean(iat) if len(iat) > 0 else 0,
        'iat_median': np.median(iat) if len(iat) > 0 else 0,
        'iat_std': np.std(iat) if len(iat) > 0 else 0,
        'iat_cv': iat_cv,

        # TLS features
        'tls_record_mean': np.mean(all_tls) if all_tls else 0,
        'tls_record_count': len(all_tls),
        'tls_cv': tls_cv,

        # Flow characteristics
        'flow_duration': flow_duration,
        'packet_rate': packet_rate,
        'large_pkt_ratio': large_pkt_ratio,
        'small_pkt_ratio': small_pkt_ratio,
    }

    return features, data


def validate_features(features):
    """Validate extracted features and show warnings"""
    print("\n" + "=" * 60)
    print("Feature Validation")
    print("=" * 60)

    warnings = []

    # Check packet counts
    total_pkts = features['pkt_count_up'] + features['pkt_count_down']
    if total_pkts < 50:
        warnings.append(f"Low packet count ({total_pkts}) - may affect accuracy")

    # Check bytes ratio
    if features['bytes_ratio'] > 100:
        warnings.append(f"Unusual bytes_ratio: {features['bytes_ratio']:.2f} (very high)")
    elif features['bytes_ratio'] < 0.1:
        warnings.append(f"Unusual bytes_ratio: {features['bytes_ratio']:.2f} (very low)")

    # Check TLS records
    if features['tls_record_mean'] == 0:
        warnings.append("No TLS records detected - this may affect classification!")

    # Check for zero values in important features
    important_zero = []
    important_features = ['pkt_cv', 'downstream_cv', 'packet_rate', 'flow_duration']
    for feat in important_features:
        if features.get(feat, 0) == 0:
            important_zero.append(feat)

    if important_zero:
        warnings.append(f"Zero-valued important features: {', '.join(important_zero)}")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"  • {w}")
    else:
        print("  ✓ All features look reasonable")

    return features


def print_feature_stats(features):
    """Print detailed feature statistics"""
    print("\n" + "=" * 60)
    print("Extracted Features")
    print("=" * 60)

    print(f"\nPacket Statistics:")
    print(f"  Upstream packets:   {features['pkt_count_up']:6d}")
    print(f"  Downstream packets: {features['pkt_count_down']:6d}")
    print(f"  Packet count ratio: {features['pkt_count_ratio']:6.2f}")
    print(f"  Mean packet size:   {features['pkt_mean_size']:6.1f} bytes")
    print(f"  Max packet size:    {features['pkt_max_size']:6d} bytes")
    print(f"  Packet size CV:     {features['pkt_cv']:6.3f}")

    print(f"\nDirectional Statistics:")
    print(f"  Downstream mean:    {features['downstream_mean_size']:6.1f} bytes")
    print(f"  Downstream CV:      {features['downstream_cv']:6.3f}")
    print(f"  Upstream mean:      {features['upstream_mean_size']:6.1f} bytes")

    print(f"\nBurst Statistics:")
    print(f"  Mean burst size:    {features['burst_mean']:6.2f} packets")
    print(f"  Max burst size:     {features['burst_max']:6d} packets")
    print(f"  Burst count:        {features['burst_count']:6d}")
    print(f"  Burst frequency:    {features['burst_frequency']:6.2f} /sec")
    print(f"  Burst CV:           {features['burst_cv']:6.3f}")

    print(f"\nTiming Statistics:")
    print(f"  Flow duration:      {features['flow_duration']:6.2f} sec")
    print(f"  Packet rate:        {features['packet_rate']:6.2f} pkts/sec")
    print(f"  IAT mean:           {features['iat_mean']:6.4f} sec")
    print(f"  IAT std dev:        {features['iat_std']:6.4f} sec")
    print(f"  IAT CV:             {features['iat_cv']:6.3f}")

    print(f"\nProtocol Statistics:")
    print(f"  Bytes ratio (D/U):  {features['bytes_ratio']:6.2f}")
    print(f"  Total bytes:        {features['total_bytes']:6d}")
    print(f"  Downstream thruput: {features['downstream_throughput']:6.1f} bytes/sec")
    print(f"  TLS record mean:    {features['tls_record_mean']:6.1f} bytes")
    print(f"  TLS record count:   {features['tls_record_count']:6d}")
    print(f"  TLS CV:             {features['tls_cv']:6.3f}")

    print(f"\nPacket Distribution:")
    print(f"  Large pkt ratio:    {features['large_pkt_ratio']:6.3f}")
    print(f"  Small pkt ratio:    {features['small_pkt_ratio']:6.3f}")

    print("=" * 60)


# -----------------------------
# Traffic Capture
# -----------------------------
def capture_traffic():
    """Capture live network traffic"""
    packets, total = [], 0

    def packet_callback(pkt):
        nonlocal total
        total += 1
        if is_spotify_packet(pkt):
            packets.append(pkt)

    print("\n" + "=" * 60)
    print(f"[*] Capturing traffic for {CAPTURE_DURATION}s on {INTERFACE}")
    print("=" * 60)
    print("\n🎧 Please play some content on Spotify now...")
    print("   (Music or Podcast - make sure it's streaming)\n")

    try:
        sniff(
            iface=INTERFACE,
            prn=packet_callback,
            timeout=CAPTURE_DURATION,
            store=False
        )
    except Exception as e:
        print(f"\n❌ Capture error: {e}")
        print("   Try running with sudo/admin privileges")
        print(f"   Or change INTERFACE from '{INTERFACE}' to your network interface")
        return []

    print(f"\n[+] Capture complete!")
    print(f"    Total packets:   {total}")
    print(f"    Spotify packets: {len(packets)}")

    if packets:
        wrpcap(CAPTURE_FILE, packets)
        print(f"    Saved to:        {CAPTURE_FILE}")
    else:
        print("\n⚠️  No Spotify traffic captured!")
        print("   Make sure Spotify is streaming and try again")

    return packets


# -----------------------------
# Prediction
# -----------------------------
def predict_traffic(features, content_features, genre_features, content_model, genre_model, label_encoder):
    """Predict content type and genre from features"""

    # Create DataFrame with exact features the model expects
    try:
        X = pd.DataFrame([features])[content_features]
    except KeyError as e:
        print(f"\n❌ ERROR: Missing feature in extracted data: {e}")
        print(f"\nExtracted features: {list(features.keys())}")
        print(f"Expected features:  {content_features}")
        missing = set(content_features) - set(features.keys())
        print(f"Missing: {missing}")
        return None

    # Check for NaN values
    if X.isnull().any().any():
        print("\n⚠️  WARNING: NaN values detected in features!")
        print(X.isnull().sum())
        X = X.fillna(0)

    result = {}

    # Predict content type
    ct = content_model.predict(X)[0]
    ct_proba = content_model.predict_proba(X)[0]

    result["content_type"] = ct
    result["content_confidence"] = max(ct_proba) * 100
    result["content_proba"] = dict(
        zip(content_model.classes_, ct_proba * 100)
    )

    # Predict genre if music
    if ct.lower() == "music":
        X_genre = pd.DataFrame([features])[genre_features]

        genre_encoded = genre_model.predict(X_genre)[0]
        genre_proba = genre_model.predict_proba(X_genre)[0]

        genre = label_encoder.inverse_transform([genre_encoded])[0]
        result["genre"] = genre
        result["genre_confidence"] = max(genre_proba) * 100

        result["top_genres"] = sorted(
            {
                label_encoder.inverse_transform([i])[0]: p * 100
                for i, p in enumerate(genre_proba)
            }.items(),
            key=lambda x: -x[1]
        )[:3]

    return result


def print_results(res):
    """Print prediction results in a nice format"""
    if not res:
        return

    print("\n" + "=" * 60)
    print("🎵 PREDICTION RESULTS")
    print("=" * 60)

    print(f"\n📊 Content Type: {res['content_type'].upper()}")
    print(f"   Confidence: {res['content_confidence']:.1f}%")

    print(f"\n   Probabilities:")
    for k, v in sorted(res["content_proba"].items(), key=lambda x: -x[1]):
        bar = "█" * int(v / 5)
        print(f"     {k:10s}: {v:5.1f}% {bar}")

    if "genre" in res:
        print(f"\n🎸 Genre: {res['genre'].upper()}")
        print(f"   Confidence: {res['genre_confidence']:.1f}%")

        print(f"\n   Top 3 Genres:")
        for i, (g, p) in enumerate(res["top_genres"], 1):
            bar = "█" * int(p / 5)
            print(f"     {i}. {g:15s}: {p:5.1f}% {bar}")

    print("\n" + "=" * 60)


# -----------------------------
# Main
# -----------------------------
def main():
    print("\n" + "=" * 60)
    print("    🎵 SPOTIFY LIVE TRAFFIC ANALYZER 🎵")
    print("=" * 60)

    # Load models
    content_model, genre_model, label_encoder, content_features, genre_features = load_models()

    # Capture traffic
    packets = capture_traffic()
    if not packets:
        print("\n❌ No packets captured. Exiting.")
        return

    # Extract features
    features, raw_data = compute_flow_features(packets)
    if not features:
        print("\n❌ Feature extraction failed")
        return

    # Print and validate features
    print_feature_stats(features)
    validate_features(features)

    # Make prediction
    print("\n" + "=" * 60)
    print("Running prediction...")
    print("=" * 60)

    results = predict_traffic(
        features,
        content_features,
        genre_features,
        content_model,
        genre_model,
        label_encoder
    )

    # Print results
    print_results(results)

    print(f"\n✅ Analysis complete!")
    print(f"   Capture saved to: {CAPTURE_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()