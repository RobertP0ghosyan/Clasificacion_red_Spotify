#!/usr/bin/env python3
"""
Enhanced PCAP Feature Extractor for Spotify Traffic Classification
Extracts temporal features and peak detection from pcap files
"""

import os
import numpy as np
import pandas as pd
from scapy.all import rdpcap
from pathlib import Path
import warnings
from scipy import stats
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

# Configuration
PCAP_ROOT = "pcap"
OUTPUT_FILE = "spotify_features.csv"


def calculate_peaks(intervals, counts, prominence_threshold=0.3):
    """
    Detect and analyze peaks in packet flow

    Parameters:
    - intervals: time intervals (seconds)
    - counts: packet counts per interval
    - prominence_threshold: minimum prominence for peak detection (relative to max)

    Returns:
    - Dictionary of peak statistics
    """
    if len(counts) == 0 or max(counts) == 0:
        return {
            'peak_count': 0,
            'peak_mean_height': 0,
            'peak_max_height': 0,
            'peak_mean_prominence': 0,
            'peak_total_area': 0,
            'peak_frequency': 0
        }

    # Normalize counts for peak detection
    max_count = max(counts)
    prominence = prominence_threshold * max_count

    # Find peaks with minimum prominence
    peaks, properties = find_peaks(counts, prominence=prominence)

    if len(peaks) == 0:
        return {
            'peak_count': 0,
            'peak_mean_height': 0,
            'peak_max_height': 0,
            'peak_mean_prominence': 0,
            'peak_total_area': 0,
            'peak_frequency': 0
        }

    # Calculate peak heights
    peak_heights = [counts[p] for p in peaks]

    # Calculate total area under peaks (approximate)
    peak_total_area = sum(peak_heights)

    # Peak frequency (peaks per second)
    duration = len(intervals)
    peak_frequency = len(peaks) / duration if duration > 0 else 0

    return {
        'peak_count': len(peaks),
        'peak_mean_height': np.mean(peak_heights),
        'peak_max_height': max(peak_heights),
        'peak_mean_prominence': np.mean(properties['prominences']),
        'peak_total_area': peak_total_area,
        'peak_frequency': peak_frequency
    }


def get_packets_per_second(packets, interval=1.0):
    """
    Calculate packet counts per time interval

    Parameters:
    - packets: list of scapy packets
    - interval: time interval in seconds (default 1.0)

    Returns:
    - intervals: array of time points
    - counts: array of packet counts per interval
    """
    if len(packets) == 0:
        return np.array([]), np.array([])

    # Get timestamps
    timestamps = [float(pkt.time) for pkt in packets]
    start_time = min(timestamps)
    end_time = max(timestamps)

    # Create intervals
    duration = int(end_time - start_time) + 1
    intervals = np.arange(0, duration, interval)
    counts = np.zeros(duration)

    # Count packets in each interval
    for ts in timestamps:
        idx = int((ts - start_time) / interval)
        if idx < len(counts):
            counts[idx] += 1

    return intervals, counts


def calculate_traffic_distribution(counts):
    """
    Calculate traffic distribution statistics

    Parameters:
    - counts: packet counts per time interval

    Returns:
    - Dictionary of distribution features
    """
    if len(counts) == 0 or sum(counts) == 0:
        return {
            'traffic_std': 0,
            'traffic_cv': 0,
            'traffic_skewness': 0,
            'traffic_kurtosis': 0,
            'active_ratio': 0,
            'peak_to_avg_ratio': 0
        }

    # Filter non-zero counts for active periods
    active_counts = counts[counts > 0]
    active_ratio = len(active_counts) / len(counts) if len(counts) > 0 else 0

    # Calculate statistics
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    cv = std_count / mean_count if mean_count > 0 else 0

    # Peak to average ratio
    max_count = max(counts)
    peak_to_avg = max_count / mean_count if mean_count > 0 else 0

    # Skewness and kurtosis (shape of distribution)
    skewness = stats.skew(counts) if len(counts) > 2 else 0
    kurtosis = stats.kurtosis(counts) if len(counts) > 3 else 0

    return {
        'traffic_std': std_count,
        'traffic_cv': cv,
        'traffic_skewness': skewness,
        'traffic_kurtosis': kurtosis,
        'active_ratio': active_ratio,
        'peak_to_avg_ratio': peak_to_avg
    }


def calculate_initial_burst(counts, window=10):
    """
    Calculate initial burst characteristics (first N seconds)

    Parameters:
    - counts: packet counts per time interval
    - window: initial window size in seconds

    Returns:
    - Dictionary of initial burst features
    """
    if len(counts) == 0:
        return {
            'initial_burst_total': 0,
            'initial_burst_avg': 0,
            'initial_burst_max': 0,
            'initial_burst_ratio': 0
        }

    # Get first N seconds
    window_size = min(window, len(counts))
    initial_window = counts[:window_size]

    initial_total = sum(initial_window)
    total_packets = sum(counts)

    return {
        'initial_burst_total': initial_total,
        'initial_burst_avg': np.mean(initial_window),
        'initial_burst_max': max(initial_window) if len(initial_window) > 0 else 0,
        'initial_burst_ratio': initial_total / total_packets if total_packets > 0 else 0
    }


def extract_features(pcap_file, genre, content_type):
    """
    Extract features from pcap file including peak detection

    Parameters:
    - pcap_file: path to pcap file
    - genre: music genre (folder name)
    - content_type: type of content (podcast/music)

    Returns:
    - Dictionary of features
    """
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"Error reading {pcap_file}: {e}")
        return None

    if len(packets) == 0:
        print(f"No packets found in {pcap_file}")
        return None

    # Get packet sizes
    pkt_sizes = [len(pkt) for pkt in packets]

    # Get timestamps
    timestamps = [float(pkt.time) for pkt in packets]

    # Calculate flow duration
    flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

    # Get packets per second
    intervals, counts = get_packets_per_second(packets, interval=1.0)

    # Calculate packet rate (packets per second)
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

    # Calculate features
    features = {
        # Basic packet statistics
        'pkt_count': len(packets),
        'pkt_avg_len': np.mean(pkt_sizes),
        'pkt_min_len': min(pkt_sizes),
        'pkt_max_len': max(pkt_sizes),
        'pkt_std_len': np.std(pkt_sizes),

        # Flow characteristics
        'flow_duration': flow_duration,
        'pkt_rate': pkt_rate,
        'burst_rate': burst_rate,

        # Peak detection features
        'peak_count': peak_features['peak_count'],
        'peak_mean_height': peak_features['peak_mean_height'],
        'peak_max_height': peak_features['peak_max_height'],
        'peak_mean_prominence': peak_features['peak_mean_prominence'],
        'peak_total_area': peak_features['peak_total_area'],
        'peak_frequency': peak_features['peak_frequency'],

        # Traffic distribution
        'traffic_std': distribution_features['traffic_std'],
        'traffic_cv': distribution_features['traffic_cv'],
        'traffic_skewness': distribution_features['traffic_skewness'],
        'traffic_kurtosis': distribution_features['traffic_kurtosis'],
        'active_ratio': distribution_features['active_ratio'],
        'peak_to_avg_ratio': distribution_features['peak_to_avg_ratio'],

        # Initial burst characteristics
        'initial_burst_total': initial_features['initial_burst_total'],
        'initial_burst_avg': initial_features['initial_burst_avg'],
        'initial_burst_max': initial_features['initial_burst_max'],
        'initial_burst_ratio': initial_features['initial_burst_ratio'],

        # Inter-arrival times
        'iat_mean': np.mean(inter_arrivals) if len(inter_arrivals) > 0 else 0,
        'iat_median': np.median(inter_arrivals) if len(inter_arrivals) > 0 else 0,
        'iat_std': np.std(inter_arrivals) if len(inter_arrivals) > 0 else 0,
        'iat_min': min(inter_arrivals) if len(inter_arrivals) > 0 else 0,
        'iat_max': max(inter_arrivals) if len(inter_arrivals) > 0 else 0,

        # Labels
        'content_type': content_type,
        'genre': genre
    }

    return features


def process_directory(root_dir):
    """
    Process all pcap files in the directory structure

    Parameters:
    - root_dir: root directory containing genre folders

    Returns:
    - DataFrame with extracted features
    """
    all_features = []

    # Map folder names to content types
    content_type_map = {
        'podcast': 'podcast',
        'rock': 'music',
        'rap': 'music',
        'edm': 'music'
    }

    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Directory '{root_dir}' not found!")
        return pd.DataFrame()

    # Iterate through genre folders
    for genre_folder in root_path.iterdir():
        if not genre_folder.is_dir():
            continue

        genre = genre_folder.name
        content_type = content_type_map.get(genre.lower(), 'unknown')

        print(f"\nProcessing genre: {genre} (content_type: {content_type})")

        # Process all pcap files in the genre folder
        pcap_files = list(genre_folder.glob('*.pcap')) + list(genre_folder.glob('*.pcapng'))

        if not pcap_files:
            print(f"  No pcap files found in {genre_folder}")
            continue

        for pcap_file in pcap_files:
            print(f"  Processing: {pcap_file.name}")
            features = extract_features(str(pcap_file), genre, content_type)

            if features:
                # Extract content_id from filename
                features['content_id'] = pcap_file.stem
                all_features.append(features)

    return pd.DataFrame(all_features)


def main():
    """Main execution function"""
    print("=" * 60)
    print("Enhanced PCAP Feature Extractor")
    print("Peak Detection & Traffic Pattern Analysis")
    print("=" * 60)

    # Process all pcap files
    df = process_directory(PCAP_ROOT)

    if df.empty:
        print("\nNo features extracted. Check if:")
        print(f"1. Directory '{PCAP_ROOT}' exists")
        print("2. Subdirectories contain .pcap files")
        return

    # Reorder columns for better readability
    label_cols = ['content_id', 'content_type', 'genre']
    feature_cols = [col for col in df.columns if col not in label_cols]
    df = df[label_cols + feature_cols]

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print(f"Feature extraction complete!")
    print(f"Total files processed: {len(df)}")
    print(f"Total features extracted: {len(feature_cols)}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"{'=' * 60}\n")

    # Display summary statistics
    print("Summary by genre:")
    print(df.groupby('genre').size())

    print("\nSummary by content type:")
    print(df.groupby('content_type').size())

    print("\nFirst few rows:")
    print(df.head())

    # Display key feature comparison
    print("\n\nKey Feature Comparison (Music vs Podcast):")
    comparison_features = [
        'pkt_count', 'pkt_rate', 'burst_rate',
        'peak_count', 'peak_frequency', 'peak_to_avg_ratio',
        'traffic_cv', 'initial_burst_ratio'
    ]

    for feature in comparison_features:
        if feature in df.columns:
            print(f"\n{feature}:")
            print(df.groupby('content_type')[feature].agg(['mean', 'std']))


if __name__ == "__main__":
    main()