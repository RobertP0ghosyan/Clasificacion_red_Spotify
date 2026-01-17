#!/usr/bin/env python3
import subprocess
import time
import sys

def run_step(description, command):
    print("\n" + "=" * 60)
    print(f"[+] {description}")
    print("=" * 60)

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"[!] Error during step: {description}")
        sys.exit(1)

    print(f"[✓] Finished: {description}")

# =========================
# PIPELINE
# =========================

if __name__ == "__main__":

    start = time.time()

    # 1️⃣ Capture traffic
    run_step(
        "Capturing Spotify traffic",
        "sudo ./venv/bin/python capture_single_spotify_flow.py"
    )

    # 2️⃣ Validate pcaps
    run_step(
        "Validating captured PCAPs",
        "python3 validate_pcaps.py"
    )

    # 3️⃣ Filter valid pcaps
    run_step(
        "Filtering valid PCAPs",
        "python3 filter_validated_pcaps.py"
    )

    # 4️⃣ Extract features
    run_step(
        "Extracting features",
        "python3 extract_features.py"
    )

    # 5️⃣ Predict content
    run_step(
        "Running prediction",
        "sudo ./venv/bin/python predict.py"
    )

    end = time.time()

    print("\n" + "=" * 60)
    print("[✓] REAL-TIME PIPELINE COMPLETED")
    print(f"[i] Total time: {end - start:.2f} seconds")
    print("=" * 60)
