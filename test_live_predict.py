#!/usr/bin/env python3
"""
Spotify Model Testing Script
Test model predictions against known ground truth labels.
Tracks accuracy and generates confusion matrices.
"""

import joblib
import pandas as pd
import os
import time
import numpy as np
from scapy.all import sniff, wrpcap
from collections import defaultdict
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
INTERFACE = "enp0s3"  # Change to your network interface
CAPTURE_DURATION = 60  # Match training capture duration
MODEL_DIR = "models"
RESULTS_DIR = "test_results"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

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
# Test Session Manager
# -----------------------------
class TestSession:
    """Manages a testing session with multiple captures."""

    def __init__(self):
        self.results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = f"{RESULTS_DIR}/test_session_{self.session_id}.csv"

    def add_result(self, test_num, true_content, true_genre, pred_content,
                   pred_genre, content_conf, genre_conf, features):
        """Add a test result to the session."""
        result = {
            'test_num': test_num,
            'true_content': true_content,
            'true_genre': true_genre if true_genre else 'N/A',
            'pred_content': pred_content,
            'pred_genre': pred_genre if pred_genre else 'N/A',
            'content_correct': true_content.lower() == pred_content.lower(),
            'genre_correct': (true_genre.lower() == pred_genre.lower()
                              if true_genre and pred_genre else None),
            'content_confidence': content_conf,
            'genre_confidence': genre_conf if genre_conf else 0,
            'num_packets': features['num_packets'],
            'flow_duration': features['flow_duration'],
            'pkt_rate': features['pkt_rate']
        }
        self.results.append(result)

        # Save incrementally
        self.save_results()

    def save_results(self):
        """Save results to CSV file."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.session_file, index=False)

    def print_summary(self):
        """Print session summary statistics."""
        if not self.results:
            print("\n❌ No results to summarize")
            return

        df = pd.DataFrame(self.results)

        print("\n" + "=" * 70)
        print("📊 TEST SESSION SUMMARY")
        print("=" * 70)

        # Content type accuracy
        content_correct = df['content_correct'].sum()
        content_total = len(df)
        content_accuracy = (content_correct / content_total) * 100

        print(f"\n📻 Content Type Classification:")
        print(f"   Accuracy: {content_accuracy:.1f}% ({content_correct}/{content_total})")
        print(f"   Avg Confidence: {df['content_confidence'].mean():.1f}%")

        # Content confusion matrix
        print(f"\n   Confusion Matrix:")
        content_matrix = pd.crosstab(
            df['true_content'],
            df['pred_content'],
            rownames=['True'],
            colnames=['Predicted']
        )
        print(content_matrix.to_string())

        # Genre accuracy (only for music)
        music_df = df[df['true_content'].str.lower() == 'music']
        if len(music_df) > 0:
            genre_results = music_df[music_df['genre_correct'].notna()]
            if len(genre_results) > 0:
                genre_correct = genre_results['genre_correct'].sum()
                genre_total = len(genre_results)
                genre_accuracy = (genre_correct / genre_total) * 100

                print(f"\n🎸 Genre Classification (Music only):")
                print(f"   Accuracy: {genre_accuracy:.1f}% ({genre_correct}/{genre_total})")
                print(f"   Avg Confidence: {genre_results['genre_confidence'].mean():.1f}%")

                # Genre confusion matrix
                print(f"\n   Confusion Matrix:")
                genre_matrix = pd.crosstab(
                    genre_results['true_genre'],
                    genre_results['pred_genre'],
                    rownames=['True'],
                    colnames=['Predicted']
                )
                print(genre_matrix.to_string())

        # Performance statistics
        print(f"\n📈 Capture Statistics:")
        print(f"   Avg packets/capture: {df['num_packets'].mean():.0f}")
        print(f"   Avg flow duration: {df['flow_duration'].mean():.1f}s")
        print(f"   Avg packet rate: {df['pkt_rate'].mean():.1f} pkt/s")

        print(f"\n💾 Results saved to: {self.session_file}")
        print("=" * 70)


# -----------------------------
# Helper Functions
# -----------------------------
def get_test_parameters():
    """Interactive prompt to get test parameters."""
    print("\n" + "=" * 70)
    print("🎯 TEST CONFIGURATION")
    print("=" * 70)

    # Content type
    print("\n1. What content type are you testing?")
    print("   [1] Music")
    print("   [2] Podcast")

    while True:
        choice = input("\n   Enter choice (1 or 2): ").strip()
        if choice == "1":
            content_type = "Music"
            break
        elif choice == "2":
            content_type = "Podcast"
            break
        else:
            print("   Invalid choice. Please enter 1 or 2.")

    # Genre (only for music)
    genre = None
    if content_type == "Music":
        print("\n2. What genre are you testing?")
        if genre_model:
            available_genres = list(genre_model.classes_)
            for i, g in enumerate(available_genres, 1):
                print(f"   [{i}] {g}")

            while True:
                try:
                    choice = int(input(f"\n   Enter choice (1-{len(available_genres)}): ").strip())
                    if 1 <= choice <= len(available_genres):
                        genre = available_genres[choice - 1]
                        break
                    else:
                        print(f"   Invalid choice. Please enter 1-{len(available_genres)}.")
                except ValueError:
                    print("   Invalid input. Please enter a number.")
        else:
            genre = input("\n   Enter genre name: ").strip()

    # Number of tests
    print("\n3. How many test captures do you want to run?")
    while True:
        try:
            num_tests = int(input("   Enter number (1-20): ").strip())
            if 1 <= num_tests <= 20:
                break
            else:
                print("   Please enter a number between 1 and 20.")
        except ValueError:
            print("   Invalid input. Please enter a number.")

    # Capture duration (optional override)
    print(f"\n4. Capture duration per test (default: {CAPTURE_DURATION}s)")
    duration_input = input(f"   Press ENTER for default or enter seconds (10-120): ").strip()

    if duration_input:
        try:
            duration = int(duration_input)
            if not (10 <= duration <= 120):
                print(f"   Invalid duration. Using default: {CAPTURE_DURATION}s")
                duration = CAPTURE_DURATION
        except ValueError:
            print(f"   Invalid input. Using default: {CAPTURE_DURATION}s")
            duration = CAPTURE_DURATION
    else:
        duration = CAPTURE_DURATION

    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    print(f"   Content Type: {content_type}")
    if genre:
        print(f"   Genre: {genre}")
    print(f"   Number of Tests: {num_tests}")
    print(f"   Capture Duration: {duration}s")
    print(f"   Interface: {INTERFACE}")
    print("=" * 70)

    confirm = input("\n✅ Start testing? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\n❌ Testing cancelled.")
        exit(0)

    return {
        'content_type': content_type,
        'genre': genre,
        'num_tests': num_tests,
        'duration': duration
    }


def capture_traffic(duration, interface=INTERFACE):
    """Capture network traffic for specified duration."""
    try:
        packets = sniff(
            iface=interface,
            filter="tcp port 443 or udp port 443",
            timeout=duration,
            store=True
        )
        return packets
    except PermissionError:
        print("[!] Permission denied. Run this script with sudo/administrator privileges.")
        exit(1)
    except Exception as e:
        print(f"[!] Capture error: {e}")
        return []


def compute_flow_features(packets):
    """Extract flow-level features from captured packets."""
    if not packets:
        return None

    current_capture = []
    last_packet_time = None

    for packet in packets:
        arrival_time = float(packet.time)
        payload_size = len(packet)
        inter_arrival = 0

        if last_packet_time is not None:
            inter_arrival = arrival_time - last_packet_time
        last_packet_time = arrival_time

        current_capture.append({
            "arrival_time": arrival_time,
            "payload_size": payload_size,
            "inter_arrival": inter_arrival
        })

    if not current_capture:
        return None

    # Extract features
    pkt_sizes = np.array([p["payload_size"] for p in current_capture])
    inter_arrivals = np.array([p["inter_arrival"] for p in current_capture[1:]])
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

    # Burst detection
    BURST_WINDOW = 0.5
    bursts = []
    window = []
    for t in timestamps:
        window = [x for x in window if t - x <= BURST_WINDOW]
        window.append(t)
        bursts.append(len(window))
    burst_mean = np.mean(bursts) if bursts else 0
    burst_max = max(bursts) if bursts else 0

    # Silence gaps
    SILENCE_THRESHOLD = 2.0
    silence_gaps = [x for x in inter_arrivals if x > SILENCE_THRESHOLD]
    num_silence_gaps = len(silence_gaps)
    silence_ratio = sum(silence_gaps) / sum(inter_arrivals) if sum(inter_arrivals) > 0 else 0

    # Flow duration
    flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    pkt_rate = len(pkt_sizes) / flow_duration if flow_duration > 0 else 0

    return {
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


def predict_traffic(features):
    """Make predictions using trained models."""
    if not features:
        return None

    feature_columns = [
        'num_packets',
        'pkt_size_mean', 'pkt_size_std', 'pkt_size_cv',
        'inter_mean', 'inter_std', 'inter_cv', 'p95_inter',
        'burst_mean', 'burst_max',
        'num_silence_gaps', 'silence_ratio',
        'pkt_rate',
        'flow_duration'
    ]

    X = pd.DataFrame([features])[feature_columns]
    results = {}

    # Predict content type
    if content_model:
        content_type = content_model.predict(X)[0]
        content_proba = content_model.predict_proba(X)[0]
        content_confidence = max(content_proba) * 100

        results["content_type"] = content_type
        results["content_confidence"] = content_confidence

        # Predict genre if music
        if content_type.lower() == "music" and genre_model:
            genre = genre_model.predict(X)[0]
            genre_proba = genre_model.predict_proba(X)[0]
            genre_confidence = max(genre_proba) * 100

            results["genre"] = genre
            results["genre_confidence"] = genre_confidence

    return results


def run_single_test(test_num, params, session):
    """Run a single test capture."""
    print("\n" + "=" * 70)
    print(f"🧪 TEST {test_num}/{params['num_tests']}")
    print("=" * 70)
    print(f"Expected: {params['content_type']}", end="")
    if params['genre']:
        print(f" - {params['genre']}")
    else:
        print()

    print(f"\n💡 Make sure Spotify is playing the correct content!")
    input("▶️  Press ENTER to start capture...")

    # Capture
    print(f"\n[+] Capturing for {params['duration']}s...")
    start_time = time.time()
    packets = capture_traffic(duration=params['duration'])
    capture_time = time.time() - start_time

    if not packets or len(packets) < 10:
        print(f"❌ Insufficient packets captured ({len(packets)}). Skipping test.")
        return False

    print(f"✓ Captured {len(packets)} packets in {capture_time:.1f}s")

    # Extract features
    print("[+] Extracting features...")
    features = compute_flow_features(packets)

    if not features:
        print("❌ Failed to extract features. Skipping test.")
        return False

    print(f"✓ Features extracted ({features['num_packets']} packets processed)")

    # Predict
    print("[+] Running prediction...")
    predictions = predict_traffic(features)

    if not predictions:
        print("❌ Prediction failed. Skipping test.")
        return False

    # Display results
    pred_content = predictions.get('content_type', 'Unknown')
    pred_genre = predictions.get('genre', None)
    content_conf = predictions.get('content_confidence', 0)
    genre_conf = predictions.get('genre_confidence', 0)

    print("\n" + "-" * 70)
    print("📊 RESULTS")
    print("-" * 70)
    print(f"Predicted Content: {pred_content} ({content_conf:.1f}% confidence)")

    # Check content correctness
    content_correct = params['content_type'].lower() == pred_content.lower()
    if content_correct:
        print("✅ Content prediction CORRECT")
    else:
        print(f"❌ Content prediction INCORRECT (expected: {params['content_type']})")

    # Genre results
    if pred_genre and params['genre']:
        print(f"Predicted Genre: {pred_genre} ({genre_conf:.1f}% confidence)")
        genre_correct = params['genre'].lower() == pred_genre.lower()
        if genre_correct:
            print("✅ Genre prediction CORRECT")
        else:
            print(f"❌ Genre prediction INCORRECT (expected: {params['genre']})")

    print("-" * 70)

    # Add to session
    session.add_result(
        test_num=test_num,
        true_content=params['content_type'],
        true_genre=params['genre'],
        pred_content=pred_content,
        pred_genre=pred_genre,
        content_conf=content_conf,
        genre_conf=genre_conf,
        features=features
    )

    return True


# -----------------------------
# Main Execution
# -----------------------------
def main():
    """Main execution flow."""
    print("\n" + "=" * 70)
    print("🧪 SPOTIFY MODEL TESTING SUITE")
    print("=" * 70)

    # Check if models are loaded
    if not content_model:
        print("\n❌ Content model not loaded! Please train models first.")
        exit(1)

    # Get test parameters
    params = get_test_parameters()

    # Create test session
    session = TestSession()

    print("\n" + "=" * 70)
    print("🚀 STARTING TEST SESSION")
    print("=" * 70)

    # Run tests
    successful_tests = 0
    for i in range(1, params['num_tests'] + 1):
        success = run_single_test(i, params, session)
        if success:
            successful_tests += 1

        # Pause between tests (except for last one)
        if i < params['num_tests']:
            print("\n⏸️  Pausing before next test...")
            time.sleep(3)

    # Print summary
    print("\n" + "=" * 70)
    print(f"✅ Testing complete! ({successful_tests}/{params['num_tests']} successful)")
    print("=" * 70)

    session.print_summary()

    print("\n✨ All done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)