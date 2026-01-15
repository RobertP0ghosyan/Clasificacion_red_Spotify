#!/usr/bin/env python3

import joblib
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scapy.all import sniff, wrpcap, rdpcap, IP
from scipy.signal import find_peaks

# -----------------------------
# Configuration
# -----------------------------
CAPTURE_FILE = "live_capture.pcap"
INTERFACE = "Wi-Fi"  # Change if needed (e.g., "eth0", "en0")
CAPTURE_DURATION = 60  # seconds
MODEL_DIR = "models"


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
        genre_features = joblib.load(f"{MODEL_DIR}/genre_lasso_features.pkl")

        print(f"  ✓ Content type model loaded")
        print(f"  ✓ Genre model loaded")
        print(f"  ✓ Label encoder loaded")
        print(f"  ✓ Feature lists loaded: {len(content_features)} content features")

        return content_model, genre_model, label_encoder, content_features, genre_features

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nMake sure you've run train.py first!")
        raise


# -----------------------------
# Traffic Capture
# -----------------------------
def capture_traffic():
    """Capture live network traffic (all IP packets)"""
    packets, total = [], 0

    def packet_callback(pkt):
        nonlocal total
        total += 1
        if pkt.haslayer(IP):
            packets.append(pkt)

    print("\n" + "=" * 60)
    print(f"[*] Capturing traffic for {CAPTURE_DURATION}s on {INTERFACE}")
    print("=" * 60)
    print("\n🎧 Capturing ALL network traffic...")
    print("   (No filtering applied)\n")

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
    print(f"    Total packets:     {total}")
    print(f"    Captured packets:  {len(packets)}")

    if packets:
        wrpcap(CAPTURE_FILE, packets)
        print(f"    Saved to:          {CAPTURE_FILE}")
    else:
        print("\n⚠️  No packets captured!")
        print("   Check your network interface and try again")

    return packets


# -----------------------------
# Feature Extraction (New Peak Detection Features)
# -----------------------------
def get_packets_per_second(packets, interval=1.0):
    """Calculate packet counts per time interval"""
    if len(packets) == 0:
        return np.array([]), np.array([])

    timestamps = [float(pkt.time) for pkt in packets]
    start_time = min(timestamps)
    end_time = max(timestamps)

    duration = int(end_time - start_time) + 1
    intervals = np.arange(0, duration, interval)
    counts = np.zeros(duration)

    for ts in timestamps:
        idx = int((ts - start_time) / interval)
        if idx < len(counts):
            counts[idx] += 1

    return intervals, counts


def calculate_peaks(intervals, counts, prominence_threshold=0.3):
    """Detect and analyze peaks in packet flow"""
    if len(counts) == 0 or max(counts) == 0:
        return {
            'peak_count': 0,
            'peak_mean_height': 0,
            'peak_max_height': 0,
            'peak_frequency': 0,
            'peak_to_avg_ratio': 0
        }

    max_count = max(counts)
    prominence = prominence_threshold * max_count

    peaks, properties = find_peaks(counts, prominence=prominence)

    if len(peaks) == 0:
        return {
            'peak_count': 0,
            'peak_mean_height': 0,
            'peak_max_height': 0,
            'peak_frequency': 0,
            'peak_to_avg_ratio': 0
        }

    peak_heights = [counts[p] for p in peaks]
    duration = len(intervals)
    peak_frequency = len(peaks) / duration if duration > 0 else 0

    avg_count = np.mean(counts[counts > 0]) if len(counts[counts > 0]) > 0 else 1
    peak_to_avg = max(peak_heights) / avg_count if avg_count > 0 else 0

    return {
        'peak_count': len(peaks),
        'peak_mean_height': np.mean(peak_heights),
        'peak_max_height': max(peak_heights),
        'peak_frequency': peak_frequency,
        'peak_to_avg_ratio': peak_to_avg
    }


def calculate_traffic_distribution(counts):
    """Calculate traffic distribution statistics"""
    if len(counts) == 0 or sum(counts) == 0:
        return {
            'traffic_std': 0,
            'traffic_cv': 0,
            'traffic_skewness': 0,
            'active_ratio': 0
        }

    from scipy import stats

    active_counts = counts[counts > 0]
    active_ratio = len(active_counts) / len(counts) if len(counts) > 0 else 0

    mean_count = np.mean(counts)
    std_count = np.std(counts)
    cv = std_count / mean_count if mean_count > 0 else 0

    skewness = stats.skew(counts) if len(counts) > 2 else 0

    return {
        'traffic_std': std_count,
        'traffic_cv': cv,
        'traffic_skewness': skewness,
        'active_ratio': active_ratio
    }


def calculate_initial_burst(counts, window=10):
    """Calculate initial burst characteristics"""
    if len(counts) == 0:
        return {
            'initial_burst_ratio': 0,
            'initial_burst_max': 0
        }

    window_size = min(window, len(counts))
    initial_window = counts[:window_size]

    initial_total = sum(initial_window)
    total_packets = sum(counts)

    return {
        'initial_burst_ratio': initial_total / total_packets if total_packets > 0 else 0,
        'initial_burst_max': max(initial_window) if len(initial_window) > 0 else 0
    }


def extract_features(packets):
    """Extract comprehensive traffic features including peak detection"""
    if not packets:
        return None

    print(f"\n[*] Extracting features from {len(packets)} packets...")

    # Get packet sizes and timestamps
    pkt_sizes = [len(pkt) for pkt in packets]
    timestamps = [float(pkt.time) for pkt in packets]

    # Sort by timestamp
    sorted_indices = np.argsort(timestamps)
    timestamps = [timestamps[i] for i in sorted_indices]

    # Calculate flow duration
    flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

    # Get packets per second
    intervals, counts = get_packets_per_second(packets, interval=1.0)

    # Calculate packet rate
    pkt_rate = len(packets) / flow_duration if flow_duration > 0 else 0

    # Calculate burst rate (using 0.1 second intervals)
    _, burst_counts = get_packets_per_second(packets, interval=0.1)
    burst_rate = np.mean(burst_counts[burst_counts > 0]) if len(burst_counts[burst_counts > 0]) > 0 else 0

    # Peak detection features
    peak_features = calculate_peaks(intervals, counts, prominence_threshold=0.3)

    # Traffic distribution features
    distribution_features = calculate_traffic_distribution(counts)

    # Initial burst features
    initial_features = calculate_initial_burst(counts, window=10)

    # Calculate inter-arrival times
    inter_arrivals = np.diff(timestamps) if len(timestamps) > 1 else []

    # Compile all features
    features = {
        # Basic packet statistics
        'pkt_count': len(packets),
        'pkt_avg_len': np.mean(pkt_sizes),
        'pkt_max_len': max(pkt_sizes),
        'pkt_std_len': np.std(pkt_sizes),
        'pkt_rate': pkt_rate,
        'burst_rate': burst_rate,

        # Peak detection features
        'peak_count': peak_features['peak_count'],
        'peak_mean_height': peak_features['peak_mean_height'],
        'peak_max_height': peak_features['peak_max_height'],
        'peak_frequency': peak_features['peak_frequency'],
        'peak_to_avg_ratio': peak_features['peak_to_avg_ratio'],

        # Traffic distribution
        'traffic_cv': distribution_features['traffic_cv'],
        'traffic_std': distribution_features['traffic_std'],
        'traffic_skewness': distribution_features['traffic_skewness'],
        'active_ratio': distribution_features['active_ratio'],

        # Initial burst characteristics
        'initial_burst_ratio': initial_features['initial_burst_ratio'],
        'initial_burst_max': initial_features['initial_burst_max'],

        # Inter-arrival times
        'iat_mean': np.mean(inter_arrivals) if len(inter_arrivals) > 0 else 0,
        'iat_std': np.std(inter_arrivals) if len(inter_arrivals) > 0 else 0,
        'iat_median': np.median(inter_arrivals) if len(inter_arrivals) > 0 else 0,
    }

    return features, intervals, counts


def print_feature_stats(features):
    """Print extracted feature statistics"""
    print("\n" + "=" * 60)
    print("Extracted Features")
    print("=" * 60)

    print(f"\nBasic Statistics:")
    print(f"  Packet count:       {features['pkt_count']:6d}")
    print(f"  Avg packet size:    {features['pkt_avg_len']:6.1f} bytes")
    print(f"  Max packet size:    {features['pkt_max_len']:6d} bytes")
    print(f"  Std packet size:    {features['pkt_std_len']:6.1f} bytes")
    print(f"  Packet rate:        {features['pkt_rate']:6.2f} pkts/sec")
    print(f"  Burst rate:         {features['burst_rate']:6.2f} pkts/0.1s")

    print(f"\nPeak Detection:")
    print(f"  Peak count:         {features['peak_count']:6d}")
    print(f"  Peak mean height:   {features['peak_mean_height']:6.2f} pkts")
    print(f"  Peak max height:    {features['peak_max_height']:6.2f} pkts")
    print(f"  Peak frequency:     {features['peak_frequency']:6.4f} /sec")
    print(f"  Peak to avg ratio:  {features['peak_to_avg_ratio']:6.2f}")

    print(f"\nTraffic Distribution:")
    print(f"  Traffic CV:         {features['traffic_cv']:6.3f}")
    print(f"  Traffic std:        {features['traffic_std']:6.2f}")
    print(f"  Traffic skewness:   {features['traffic_skewness']:6.3f}")
    print(f"  Active ratio:       {features['active_ratio']:6.3f}")

    print(f"\nInitial Burst:")
    print(f"  Burst ratio:        {features['initial_burst_ratio']:6.3f}")
    print(f"  Burst max:          {features['initial_burst_max']:6.2f} pkts")

    print(f"\nInter-Arrival Times:")
    print(f"  IAT mean:           {features['iat_mean']:6.4f} sec")
    print(f"  IAT std:            {features['iat_std']:6.4f} sec")
    print(f"  IAT median:         {features['iat_median']:6.4f} sec")

    print("=" * 60)


def plot_packets_over_time(intervals, counts, output_file="packet_timeline.png"):
    """Create packets vs time graph with 1-second bins"""
    if len(counts) == 0:
        print("No data to plot")
        return

    print("\n" + "=" * 60)
    print("Creating packet timeline visualization...")
    print("=" * 60)

    # Create the plot
    plt.figure(figsize=(14, 7))

    # Main plot: packet counts
    ax1 = plt.subplot(2, 1, 1)
    plt.bar(intervals, counts, width=0.9, align='edge',
            color='#1DB954', edgecolor='black', alpha=0.7, label='Packets')

    # Detect and mark peaks
    if max(counts) > 0:
        peaks, _ = find_peaks(counts, prominence=0.3 * max(counts))
        if len(peaks) > 0:
            plt.plot(peaks, counts[peaks], "rx", markersize=10,
                     markeredgewidth=2, label='Detected Peaks')

    plt.xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    plt.ylabel('Packet Count', fontsize=11, fontweight='bold')
    plt.title('Packet Distribution Over Time (1-second bins)',
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()

    # Statistics
    total_packets = int(sum(counts))
    duration = len(counts)
    avg_rate = total_packets / duration if duration > 0 else 0
    max_rate = int(max(counts)) if len(counts) > 0 else 0

    stats_text = f'Total: {total_packets} pkts\n'
    stats_text += f'Duration: {duration}s\n'
    stats_text += f'Avg: {avg_rate:.1f} pkt/s\n'
    stats_text += f'Peak: {max_rate} pkt/s'

    plt.text(0.98, 0.97, stats_text,
             transform=ax1.transAxes,
             fontsize=9,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # Subplot 2: Moving average
    ax2 = plt.subplot(2, 1, 2)

    # Calculate moving average (window=5)
    window = min(5, len(counts))
    moving_avg = np.convolve(counts, np.ones(window) / window, mode='valid')

    plt.plot(range(len(moving_avg)), moving_avg,
             color='#1DB954', linewidth=2, label=f'{window}s Moving Average')
    plt.fill_between(range(len(moving_avg)), moving_avg,
                     alpha=0.3, color='#1DB954')

    plt.xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    plt.ylabel('Avg Packet Count', fontsize=11, fontweight='bold')
    plt.title('Traffic Pattern (Smoothed)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved packet timeline: {output_file}")
    print(f"  Total packets: {total_packets}")
    print(f"  Duration: {duration} seconds")
    print(f"  Average rate: {avg_rate:.2f} packets/second")
    print(f"  Peak rate: {max_rate} packets/second")


# -----------------------------
# Prediction
# -----------------------------
def predict_traffic(features, content_features, genre_features,
                    content_model, genre_model, label_encoder):
    """Predict content type and genre from features"""

    try:
        X = pd.DataFrame([features])[content_features]
    except KeyError as e:
        print(f"\n❌ ERROR: Missing feature: {e}")
        print(f"\nExtracted: {list(features.keys())}")
        print(f"Expected:  {content_features}")
        missing = set(content_features) - set(features.keys())
        print(f"Missing:   {missing}")
        return None

    if X.isnull().any().any():
        print("\n⚠️  WARNING: NaN values detected!")
        X = X.fillna(0)

    result = {}

    # Predict content type
    ct = content_model.predict(X)[0]
    ct_proba = content_model.predict_proba(X)[0]

    result["content_type"] = ct
    result["content_confidence"] = max(ct_proba) * 100
    result["content_proba"] = dict(zip(content_model.classes_, ct_proba * 100))

    # Predict genre if music
    if ct.lower() == "music":
        X_genre = pd.DataFrame([features])[genre_features]
        genre_encoded = genre_model.predict(X_genre)[0]
        genre_proba = genre_model.predict_proba(X_genre)[0]

        genre = label_encoder.inverse_transform([genre_encoded])[0]
        result["genre"] = genre
        result["genre_confidence"] = max(genre_proba) * 100

        result["top_genres"] = sorted(
            {label_encoder.inverse_transform([i])[0]: p * 100
             for i, p in enumerate(genre_proba)}.items(),
            key=lambda x: -x[1]
        )[:3]

    return result


def print_results(res):
    """Print prediction results"""
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
    print("    With Peak Detection & Pattern Analysis")
    print("=" * 60)

    # Load models
    content_model, genre_model, label_encoder, content_features, genre_features = load_models()

    # Capture traffic
    packets = capture_traffic()
    if not packets:
        print("\n❌ No packets captured. Exiting.")
        return

    # Extract features
    result = extract_features(packets)
    if not result:
        print("\n❌ Feature extraction failed")
        return

    features, intervals, counts = result

    # Print feature stats
    print_feature_stats(features)

    # Plot packet timeline
    plot_packets_over_time(intervals, counts)

    # Make prediction
    print("\n" + "=" * 60)
    print("Running prediction...")
    print("=" * 60)

    results = predict_traffic(
        features, content_features, genre_features,
        content_model, genre_model, label_encoder
    )

    # Print results
    print_results(results)

    print(f"\n✅ Analysis complete!")
    print(f"   Capture saved to: {CAPTURE_FILE}")
    print(f"   Graph saved to: packet_timeline.png")
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