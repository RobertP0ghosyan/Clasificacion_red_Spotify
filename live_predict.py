#!/usr/bin/env python3
"""
Spotify Live Traffic Analyzer and Predictor
Captures network traffic and predicts content type and genre
using the same feature extraction as the training dataset.
"""

import joblib
import pandas as pd
import os
import time
import numpy as np
from scapy.all import sniff, wrpcap

# -----------------------------
# Configuration
# -----------------------------
CAPTURE_FILE = "live_capture.pcap"
INTERFACE = "enp0s3"  # Change to your network interface
CAPTURE_DURATION = 60  # Match training capture duration
MODEL_DIR = "models"

# Load trained models
print("Loading trained models...")
try:
    content_model = joblib.load(f"{MODEL_DIR}/content_type_model.pkl")
    print("  ✓ Content type model loaded")
except FileNotFoundError:
    print("  ✗ Content type model not found!")
    content_model = None

try:
    genre_model = joblib.load(f"{MODEL_DIR}/genre_model.pkl")
    print("  ✓ Genre model loaded")
except FileNotFoundError:
    print("  ✗ Genre model not found!")
    genre_model = None


# -----------------------------
# Helper Functions
# -----------------------------
def capture_traffic(duration=CAPTURE_DURATION, interface=INTERFACE):
    """
    Capture network traffic using Scapy for a fixed duration.
    Filters for HTTPS traffic (port 443) like the training script.
    """
    print(f"\n[+] Capturing traffic for {duration} seconds on interface '{interface}'...")
    print("    Filtering: TCP port 443 (HTTPS - Spotify streaming)")

    try:
        packets = sniff(
            iface=interface,
            filter="tcp port 443",  # Same filter as training
            timeout=duration,
            store=True
        )

        if packets:
            wrpcap(CAPTURE_FILE, packets)
            print(f"[+] Capture complete! {len(packets)} packets saved to {CAPTURE_FILE}")
            return packets
        else:
            print("[!] No packets captured. Make sure Spotify is streaming.")
            return []

    except PermissionError:
        print("[!] Permission denied. Run this script with sudo/administrator privileges.")
        exit(1)
    except Exception as e:
        print(f"[!] Capture error: {e}")
        exit(1)


def compute_flow_features(packets):
    """
    Extract flow-level features from captured packets.
    MATCHES THE EXACT FEATURE COMPUTATION FROM THE TRAINING SCRIPT.
    """
    print(f"\n[+] Computing flow features from {len(packets)} packets...")

    # Store packet-level data (same as training script)
    current_capture = []
    last_packet_time = None

    for packet in packets:
        arrival_time = float(packet.time)
        payload_size = len(packet)

        # Compute inter-arrival time
        inter_arrival = 0
        if last_packet_time is not None:
            inter_arrival = arrival_time - last_packet_time
        last_packet_time = arrival_time

        # ✅ FIXED: Actually append to current_capture!
        current_capture.append({
            "arrival_time": arrival_time,
            "payload_size": payload_size,
            "inter_arrival": inter_arrival
        })

    if not current_capture:
        print("[!] No packets to process!")
        return None

    print(f"    Processed {len(current_capture)} packets")

    # ========================================
    # FLOW FEATURE COMPUTATION
    # (Exact copy from training script)
    # ========================================

    # Extract relevant lists
    pkt_sizes = np.array([p["payload_size"] for p in current_capture])
    inter_arrivals = np.array([p["inter_arrival"] for p in current_capture[1:]])  # skip first
    timestamps = np.array([p["arrival_time"] for p in current_capture])

    # Packet size features
    pkt_size_mean = np.mean(pkt_sizes)
    pkt_size_std = np.std(pkt_sizes)
    pkt_size_cv = pkt_size_std / pkt_size_mean if pkt_size_mean else 0

    # Inter-packet arrival features
    inter_mean = np.mean(inter_arrivals) if len(inter_arrivals) > 0 else 0
    inter_std = np.std(inter_arrivals) if len(inter_arrivals) > 0 else 0
    inter_cv = inter_std / inter_mean if inter_mean else 0
    p95_inter = np.percentile(inter_arrivals, 95) if len(inter_arrivals) > 0 else 0

    # Burst detection (packets within 0.5s)
    BURST_WINDOW = 0.5
    bursts = []
    window = []
    for t in timestamps:
        window = [x for x in window if t - x <= BURST_WINDOW]
        window.append(t)
        bursts.append(len(window))
    burst_mean = np.mean(bursts) if bursts else 0
    burst_max = max(bursts) if bursts else 0

    # Silence gaps (>2s)
    SILENCE_THRESHOLD = 2.0
    silence_gaps = [x for x in inter_arrivals if x > SILENCE_THRESHOLD]
    num_silence_gaps = len(silence_gaps)
    silence_ratio = sum(silence_gaps) / sum(inter_arrivals) if sum(inter_arrivals) > 0 else 0

    # Flow duration
    flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

    # Packet rate
    pkt_rate = len(pkt_sizes) / flow_duration if flow_duration > 0 else 0

    # Return feature dictionary (same order as training)
    features = {
        "num_packets": len(current_capture),
        "pkt_size_mean": pkt_size_mean,
        "pkt_size_std": pkt_size_std,
        "pkt_size_cv": pkt_size_cv,
        "inter_mean": inter_mean,
        "inter_std": inter_std,
        "inter_cv": inter_cv,
        "p95_inter": p95_inter,
        "burst_mean": burst_mean,
        "burst_max": burst_max,
        "num_silence_gaps": num_silence_gaps,
        "silence_ratio": silence_ratio,
        "flow_duration": flow_duration,
        "pkt_rate": pkt_rate
    }

    return features


def print_features(features):
    """Print extracted features in a readable format."""
    print("\n" + "=" * 60)
    print("EXTRACTED FLOW FEATURES")
    print("=" * 60)

    print("\n📦 Packet Features:")
    print(f"  Total packets:    {features['num_packets']} packets")
    print(f"  Mean size:        {features['pkt_size_mean']:.2f} bytes")
    print(f"  Std dev:          {features['pkt_size_std']:.2f} bytes")
    print(f"  Coeff. variation: {features['pkt_size_cv']:.4f}")

    print("\n⏱  Inter-Arrival Time Features:")
    print(f"  Mean interval:    {features['inter_mean']:.4f} seconds")
    print(f"  Std dev:          {features['inter_std']:.4f} seconds")
    print(f"  Coeff. variation: {features['inter_cv']:.4f}")
    print(f"  95th percentile:  {features['p95_inter']:.4f} seconds")

    print("\n💥 Burst Features:")
    print(f"  Mean burst size:  {features['burst_mean']:.2f} packets/0.5s")
    print(f"  Max burst size:   {features['burst_max']:.0f} packets/0.5s")

    print("\n🔇 Silence Features:")
    print(f"  Silence gaps:     {features['num_silence_gaps']:.0f} gaps > 2s")
    print(f"  Silence ratio:    {features['silence_ratio']:.4f}")

    print("\n📊 Flow Statistics:")
    print(f"  Flow duration:    {features['flow_duration']:.2f} seconds")
    print(f"  Packet rate:      {features['pkt_rate']:.2f} packets/sec")
    print("=" * 60)


def predict_traffic(features):
    """
    Make predictions using trained models.
    Returns dictionary with predictions and confidence scores.
    """
    if not features:
        return {"error": "No features extracted"}

    # Define feature columns in exact order from training
    feature_columns = [
        'num_packets',
        'pkt_size_mean', 'pkt_size_std', 'pkt_size_cv',
        'inter_mean', 'inter_std', 'inter_cv', 'p95_inter',
        'burst_mean', 'burst_max',
        'num_silence_gaps', 'silence_ratio',
        'pkt_rate',
        'flow_duration'
    ]

    # Convert features to DataFrame with only the feature columns
    X = pd.DataFrame([features])[feature_columns]

    results = {}

    # Predict content type (Music vs Podcast)
    if content_model:
        content_type = content_model.predict(X)[0]
        content_proba = content_model.predict_proba(X)[0]
        content_confidence = max(content_proba) * 100

        results["content_type"] = content_type
        results["content_confidence"] = content_confidence

        # Show probabilities for all classes
        results["content_proba"] = {
            cls: prob * 100
            for cls, prob in zip(content_model.classes_, content_proba)
        }

        # Predict genre only if it's music
        if content_type.lower() == "music" and genre_model:
            genre = genre_model.predict(X)[0]
            genre_proba = genre_model.predict_proba(X)[0]
            genre_confidence = max(genre_proba) * 100

            results["genre"] = genre
            results["genre_confidence"] = genre_confidence

            # Show top 3 genre probabilities
            genre_probs = {
                cls: prob * 100
                for cls, prob in zip(genre_model.classes_, genre_proba)
            }
            top_3_genres = sorted(genre_probs.items(), key=lambda x: -x[1])[:3]
            results["top_genres"] = top_3_genres

    return results


def print_predictions(results):
    """Print prediction results in a nice format."""
    print("\n" + "=" * 60)
    print("🎵 PREDICTION RESULTS")
    print("=" * 60)

    if "error" in results:
        print(f"\n❌ {results['error']}")
        return

    # Content Type
    if "content_type" in results:
        print(f"\n📻 Content Type: {results['content_type'].upper()}")
        print(f"   Confidence: {results['content_confidence']:.1f}%")

        if "content_proba" in results:
            print("   Probabilities:")
            for cls, prob in sorted(results['content_proba'].items(), key=lambda x: -x[1]):
                bar = "█" * int(prob / 5)  # Simple progress bar
                print(f"     {cls:10s}: {prob:5.1f}% {bar}")

    # Genre (only for music)
    if "genre" in results:
        print(f"\n🎸 Genre: {results['genre'].upper()}")
        print(f"   Confidence: {results['genre_confidence']:.1f}%")

        if "top_genres" in results:
            print("   Top 3 Genres:")
            for genre, prob in results["top_genres"]:
                bar = "█" * int(prob / 5)
                print(f"     {genre:20s}: {prob:5.1f}% {bar}")

    print("=" * 60)


# -----------------------------
# Main Execution
# -----------------------------
def main():
    """Main execution flow."""
    print("\n" + "=" * 60)
    print("🎵 SPOTIFY LIVE TRAFFIC ANALYZER")
    print("=" * 60)

    # Check if models are loaded
    if not content_model and not genre_model:
        print("\n❌ No models loaded! Please train models first.")
        print("   Run: python train_classification_models.py")
        exit(1)

    print("\n" + "=" * 60)
    print("STARTING LIVE CAPTURE")
    print("=" * 60)
    print(f"Duration: {CAPTURE_DURATION} seconds")
    print(f"Interface: {INTERFACE}")
    print("Filter: TCP port 443 (HTTPS)")
    print("\n💡 TIP: Start playing something on Spotify now!")

    input("\n▶️  Press ENTER to start capture...")

    # Capture traffic
    start_time = time.time()
    packets = capture_traffic(duration=CAPTURE_DURATION, interface=INTERFACE)
    capture_time = time.time() - start_time

    if not packets:
        print("\n❌ No packets captured. Make sure:")
        print("   1. Spotify is actively streaming")
        print("   2. You're using the correct network interface")
        print("   3. You have sudo/admin privileges")
        exit(1)

    print(f"⏱️  Actual capture time: {capture_time:.2f} seconds")

    # Extract features
    features = compute_flow_features(packets)

    if not features:
        print("\n❌ Failed to extract features. Exiting.")
        exit(1)

    # Display features
    print_features(features)

    # Make predictions
    print("\n🤖 Running predictions...")
    results = predict_traffic(features)

    # Display results
    print_predictions(results)

    print("\n✅ Analysis complete!")
    print(f"📁 Capture saved to: {CAPTURE_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Capture interrupted by user")
        exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)