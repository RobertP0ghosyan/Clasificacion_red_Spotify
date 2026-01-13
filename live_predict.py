#!/usr/bin/env python3

import joblib
import pandas as pd
import os
import time
import json
import numpy as np
from scapy.all import sniff, wrpcap, IP

# -----------------------------
# Configuration
# -----------------------------
CAPTURE_FILE = "live_capture.pcap"
INTERFACE = "enp0s3"  # Change to your network interface
CAPTURE_DURATION = 60  # Duration in seconds
MODEL_DIR = "models"
IP_CACHE_FILE = "spotify_ips.json"  # Cache discovered IPs

# -----------------------------
# Load cached Spotify IPs
# -----------------------------
try:
    with open(IP_CACHE_FILE, 'r') as f:
        cached_data = json.load(f)
        SPOTIFY_IPS = set(cached_data.get('ips', []))
        print(f"[*] Loaded {len(SPOTIFY_IPS)} cached Spotify IPs from {IP_CACHE_FILE}")
except FileNotFoundError:
    SPOTIFY_IPS = set()
    print(f"[*] No cached IPs found. Will use fallback IP prefixes.")

# Fallback Spotify IP prefixes
SPOTIFY_IP_PREFIXES = [
    "35.186.", "104.154.", "35.184.",
    "35.185.", "104.199.", "34.120.", "34.117."
]

# -----------------------------
# Load trained models
# -----------------------------
print("\nLoading trained models...")
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
# Spotify Traffic Filtering
# -----------------------------
def is_spotify_packet(packet):
    """Check if packet belongs to Spotify traffic."""
    if not packet.haslayer(IP):
        return False

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst

    if SPOTIFY_IPS:
        if src_ip in SPOTIFY_IPS or dst_ip in SPOTIFY_IPS:
            return True

    for prefix in SPOTIFY_IP_PREFIXES:
        if src_ip.startswith(prefix) or dst_ip.startswith(prefix):
            return True

    return False

# -----------------------------
# Feature Extraction
# -----------------------------
def calculate_bursts(timestamps, burst_gap=0.1):
    if len(timestamps) <= 1:
        return [1]

    bursts = []
    current_burst = 1
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        if gap < burst_gap:
            current_burst += 1
        else:
            bursts.append(current_burst)
            current_burst = 1
    bursts.append(current_burst)
    return bursts

def compute_flow_features(packets):
    """Compute directional and TLS features from captured packets"""
    if not packets:
        print("    Warning: No packets captured!")
        return {
            "pkt_mean_size": 0, "pkt_max_size": 0,
            "pkt_count_up": 0, "pkt_count_down": 0,
            "burst_mean": 0, "burst_max": 0,
            "bytes_ratio": 0, "iat_std": 0, "tls_record_mean": 0
        }

    # Convert scapy packets to dict with size, timestamp, direction, tls_records
    packet_dicts = []
    for p in packets:
        size = len(p)
        ts = float(p.time)
        direction = "upstream" if p[IP].src.startswith("192.") else "downstream"
        # Simplified TLS record extraction
        try:
            payload = bytes(p.payload)
            tls_records = []
            offset = 0
            while offset + 5 <= len(payload):
                content_type = payload[offset]
                if content_type not in [0x14, 0x15, 0x16, 0x17, 0x18]:
                    break
                record_length = (payload[offset + 3] << 8) | payload[offset + 4]
                tls_records.append(record_length)
                offset += 5 + record_length
        except:
            tls_records = []

        packet_dicts.append({
            "size": size,
            "timestamp": ts,
            "direction": direction,
            "tls_records": tls_records
        })

    all_sizes = [p["size"] for p in packet_dicts]
    upstream = [p["size"] for p in packet_dicts if p["direction"] == 'upstream']
    downstream = [p["size"] for p in packet_dicts if p["direction"] == 'downstream']
    timestamps = [p["timestamp"] for p in packet_dicts]
    tls_records = [r for p in packet_dicts for r in p["tls_records"]]
    inter_arrivals = np.diff(timestamps) if len(timestamps) > 1 else []
    bursts = calculate_bursts(timestamps)

    upstream_bytes = sum(upstream)
    downstream_bytes = sum(downstream)
    bytes_ratio = downstream_bytes / upstream_bytes if upstream_bytes > 0 else 0

    return {
        "pkt_mean_size": np.mean(all_sizes) if all_sizes else 0,
        "pkt_max_size": max(all_sizes) if all_sizes else 0,
        "pkt_count_up": len(upstream),
        "pkt_count_down": len(downstream),
        "burst_mean": np.mean(bursts) if bursts else 0,
        "burst_max": max(bursts) if bursts else 0,
        "bytes_ratio": bytes_ratio,
        "iat_std": np.std(inter_arrivals) if len(inter_arrivals) > 0 else 0,
        "tls_record_mean": np.mean(tls_records) if tls_records else 0
    }

# -----------------------------
# Traffic Capture
# -----------------------------
def capture_traffic(duration=CAPTURE_DURATION, interface=INTERFACE):
    captured_packets = []
    total_packets = 0

    def packet_callback(packet):
        nonlocal total_packets
        total_packets += 1
        if is_spotify_packet(packet):
            captured_packets.append(packet)

    print(f"\n[*] Capturing Spotify traffic for {duration} seconds on {interface}...")
    sniff(iface=interface, prn=packet_callback, timeout=duration, store=False)

    if captured_packets:
        wrpcap(CAPTURE_FILE, captured_packets)
        print(f"[+] Capture complete! {len(captured_packets)}/{total_packets} packets saved to {CAPTURE_FILE}")
    else:
        print(f"[!] No Spotify packets captured (saw {total_packets} total packets)")

    return captured_packets

# -----------------------------
# Prediction
# -----------------------------
def predict_traffic(features):
    if not features:
        return {"error": "No features extracted"}

    feature_columns = [
        "pkt_mean_size", "pkt_max_size",
        "pkt_count_up", "pkt_count_down",
        "burst_mean", "burst_max",
        "bytes_ratio", "iat_std", "tls_record_mean"
    ]
    X = pd.DataFrame([features])[feature_columns]
    results = {}

    if content_model:
        content_type = content_model.predict(X)[0]
        content_proba = content_model.predict_proba(X)[0]
        content_confidence = max(content_proba) * 100

        results["content_type"] = content_type
        results["content_confidence"] = content_confidence
        results["content_proba"] = {
            cls: prob * 100 for cls, prob in zip(content_model.classes_, content_proba)
        }

        if content_type.lower() == "music" and genre_model:
            genre = genre_model.predict(X)[0]
            genre_proba = genre_model.predict_proba(X)[0]
            genre_confidence = max(genre_proba) * 100

            results["genre"] = genre
            results["genre_confidence"] = genre_confidence
            genre_probs = {
                cls: prob * 100 for cls, prob in zip(genre_model.classes_, genre_proba)
            }
            results["top_genres"] = sorted(genre_probs.items(), key=lambda x: -x[1])[:3]

    return results

def print_predictions(results):
    print("\n" + "="*50)
    print("🎵 PREDICTION RESULTS")
    print("="*50)
    if "error" in results:
        print(f"\n {results['error']}")
        return
    if "content_type" in results:
        print(f"\n Content Type: {results['content_type'].upper()}")
        print(f"   Confidence: {results['content_confidence']:.1f}%")
        if "content_proba" in results:
            for cls, prob in sorted(results['content_proba'].items(), key=lambda x: -x[1]):
                print(f"     {cls:10s}: {prob:5.1f}%")
    if "genre" in results:
        print(f"\n🎸 Genre: {results['genre'].upper()}")
        print(f"   Confidence: {results['genre_confidence']:.1f}%")
        if "top_genres" in results:
            print("   Top 3 Genres:")
            for genre, prob in results["top_genres"]:
                print(f"     {genre:20s}: {prob:5.1f}%")
    print("="*50)

# -----------------------------
# Main
# -----------------------------
def main():
    print("\n🎵 SPOTIFY LIVE TRAFFIC ANALYZER (AUTOMATIC)")
    if not content_model and not genre_model:
        print("\n❌ No models loaded! Exiting.")
        return

    packets = capture_traffic()
    if not packets:
        print("\n❌ No packets captured. Exiting.")
        return

    features = compute_flow_features(packets)
    results = predict_traffic(features)
    print_predictions(results)
    print(f"\n✅ Capture saved to {CAPTURE_FILE}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    import sys
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--discover":
            from time import sleep
            print("\nStarting Spotify IP discovery mode...")
            sleep(1)
            # discovery function can be added here if needed
            print("Discovery mode not implemented in auto-run version.")
        else:
            main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
