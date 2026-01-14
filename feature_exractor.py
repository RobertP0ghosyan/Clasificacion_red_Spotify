#!/usr/bin/env python3
"""
Enhanced PCAP Feature Extractor for Spotify Traffic Classification
Extracts directional, TLS, and temporal features from pcap files
"""

import os
import json
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, Raw
from pathlib import Path
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Configuration
SPOTIFY_IPS_FILE = "spotify_ips.json"
PCAP_ROOT = "pcap"
OUTPUT_FILE = "spotify_features.csv"


def load_spotify_ips():
    """Load Spotify IPs from JSON file"""
    try:
        with open(SPOTIFY_IPS_FILE, 'r') as f:
            data = json.load(f)
            return data['ips']
    except Exception as e:
        print(f"Error loading {SPOTIFY_IPS_FILE}: {e}")
        print("Using default IP")
        return ["35.186.224.26"]


def is_tls_handshake(payload):
    """Check if payload contains TLS handshake (content type 0x16)"""
    if len(payload) < 5:
        return False
    return payload[0] == 0x16


def extract_tls_record_sizes(payload):
    """Extract TLS record sizes from payload"""
    records = []
    offset = 0

    while offset + 5 <= len(payload):
        content_type = payload[offset]
        # TLS content types: 0x14-0x18
        if content_type not in [0x14, 0x15, 0x16, 0x17, 0x18]:
            break

        # Record length is bytes 3-4 (big endian)
        record_length = (payload[offset + 3] << 8) | payload[offset + 4]
        records.append(record_length)

        # Move to next record (5 byte header + record length)
        offset += 5 + record_length

        if offset >= len(payload):
            break

    return records


def calculate_bursts(timestamps, burst_gap=0.1):
    """
    Calculate burst sizes (consecutive packets with gap < burst_gap seconds)

    Parameters:
    - timestamps: list of packet timestamps
    - burst_gap: maximum gap between packets in a burst (default 0.1 seconds)

    Returns:
    - List of burst sizes
    """
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

    # Add last burst
    bursts.append(current_burst)

    return bursts


def extract_features(pcap_file, genre, content_type, spotify_ips):
    """
    Extract directional, TLS, and temporal features from pcap file

    Parameters:
    - pcap_file: path to pcap file
    - genre: music genre (folder name)
    - content_type: type of content (podcast/music)
    - spotify_ips: list of Spotify IP addresses

    Returns:
    - Dictionary of features
    """
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"Error reading {pcap_file}: {e}")
        return None

    # Filter packets with any Spotify IP (source or destination)
    filtered_pkts = [pkt for pkt in packets if IP in pkt and
                     (pkt[IP].src in spotify_ips or pkt[IP].dst in spotify_ips)]

    if len(filtered_pkts) == 0:
        print(f"No packets found with Spotify IPs in {pcap_file}")
        return None

    # Separate upstream and downstream packets
    upstream_pkts = []  # Client to Spotify
    downstream_pkts = []  # Spotify to Client
    all_pkt_sizes = []
    upstream_sizes = []
    downstream_sizes = []
    inter_arrivals = []
    tls_record_sizes = []
    timestamps = []

    for pkt in filtered_pkts:
        pkt_size = len(pkt)
        all_pkt_sizes.append(pkt_size)
        timestamps.append(float(pkt.time))

        # Determine direction
        if pkt[IP].dst in spotify_ips:
            # Client to server (upstream)
            upstream_pkts.append(pkt_size)
            upstream_sizes.append(pkt_size)
        else:
            # Server to client (downstream)
            downstream_pkts.append(pkt_size)
            downstream_sizes.append(pkt_size)

        # Extract TLS record sizes
        if Raw in pkt:
            payload = bytes(pkt[Raw].load)
            if is_tls_handshake(payload) or (len(payload) >= 5 and payload[0] in [0x14, 0x15, 0x16, 0x17, 0x18]):
                records = extract_tls_record_sizes(payload)
                tls_record_sizes.extend(records)

    # Calculate inter-arrival times
    if len(timestamps) > 1:
        inter_arrivals = np.diff(timestamps)

    # Calculate burst statistics
    bursts = calculate_bursts(timestamps, burst_gap=0.1)

    # Flow duration and packet rate
    flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    packet_rate = len(filtered_pkts) / flow_duration if flow_duration > 0 else 0

    # Calculate bytes statistics
    upstream_bytes = sum(upstream_pkts)
    downstream_bytes = sum(downstream_pkts)
    bytes_ratio = downstream_bytes / upstream_bytes if upstream_bytes > 0 else 0
    total_bytes = upstream_bytes + downstream_bytes

    # Packet count ratio
    pkt_count_ratio = len(downstream_pkts) / len(upstream_pkts) if len(upstream_pkts) > 0 else 0

    # Coefficient of Variation for packet sizes (measure of consistency)
    pkt_cv = np.std(all_pkt_sizes) / np.mean(all_pkt_sizes) if len(all_pkt_sizes) > 0 and np.mean(
        all_pkt_sizes) > 0 else 0

    # Separate CV for downstream (music should be more consistent)
    downstream_cv = np.std(downstream_sizes) / np.mean(downstream_sizes) if len(downstream_sizes) > 0 and np.mean(
        downstream_sizes) > 0 else 0

    # Burst statistics
    burst_count = len(bursts)
    burst_frequency = burst_count / flow_duration if flow_duration > 0 else 0
    burst_cv = np.std(bursts) / np.mean(bursts) if len(bursts) > 0 and np.mean(bursts) > 0 else 0

    # Inter-arrival time statistics
    iat_mean = np.mean(inter_arrivals) if len(inter_arrivals) > 0 else 0
    iat_median = np.median(inter_arrivals) if len(inter_arrivals) > 0 else 0
    iat_cv = np.std(inter_arrivals) / np.mean(inter_arrivals) if len(inter_arrivals) > 0 and iat_mean > 0 else 0

    # TLS record statistics
    tls_cv = np.std(tls_record_sizes) / np.mean(tls_record_sizes) if len(tls_record_sizes) > 0 and np.mean(
        tls_record_sizes) > 0 else 0

    # Large packet ratio (packets > 1400 bytes, typical for full data packets)
    large_pkt_count = sum(1 for size in downstream_sizes if size > 1400)
    large_pkt_ratio = large_pkt_count / len(downstream_sizes) if len(downstream_sizes) > 0 else 0

    # Small packet ratio (packets < 100 bytes, typical for ACKs)
    small_pkt_count = sum(1 for size in all_pkt_sizes if size < 100)
    small_pkt_ratio = small_pkt_count / len(all_pkt_sizes) if len(all_pkt_sizes) > 0 else 0

    # Downstream throughput (bytes per second)
    downstream_throughput = downstream_bytes / flow_duration if flow_duration > 0 else 0

    # Calculate features
    features = {
        # Basic packet statistics
        'pkt_mean_size': np.mean(all_pkt_sizes) if all_pkt_sizes else 0,
        'pkt_max_size': max(all_pkt_sizes) if all_pkt_sizes else 0,
        'pkt_min_size': min(all_pkt_sizes) if all_pkt_sizes else 0,
        'pkt_std_size': np.std(all_pkt_sizes) if all_pkt_sizes else 0,
        'pkt_cv': pkt_cv,  # NEW: Coefficient of variation

        # Directional packet counts
        'pkt_count_up': len(upstream_pkts),
        'pkt_count_down': len(downstream_pkts),
        'pkt_count_ratio': pkt_count_ratio,  # NEW: Down/Up packet ratio

        # Directional packet sizes
        'downstream_mean_size': np.mean(downstream_sizes) if downstream_sizes else 0,
        'downstream_cv': downstream_cv,  # NEW: Downstream size consistency
        'upstream_mean_size': np.mean(upstream_sizes) if upstream_sizes else 0,

        # Burst characteristics
        'burst_mean': np.mean(bursts) if bursts else 0,
        'burst_max': max(bursts) if bursts else 0,
        'burst_count': burst_count,  # NEW: Number of bursts
        'burst_frequency': burst_frequency,  # NEW: Bursts per second
        'burst_cv': burst_cv,  # NEW: Burst consistency

        # Bytes and throughput
        'bytes_ratio': bytes_ratio,
        'total_bytes': total_bytes,  # NEW: Total data transferred
        'downstream_throughput': downstream_throughput,  # NEW: Bytes/sec downstream

        # Inter-arrival times
        'iat_mean': iat_mean,  # NEW: Mean inter-arrival time
        'iat_median': iat_median,  # NEW: Median inter-arrival time
        'iat_std': np.std(inter_arrivals) if len(inter_arrivals) > 0 else 0,
        'iat_cv': iat_cv,  # NEW: IAT consistency

        # TLS features
        'tls_record_mean': np.mean(tls_record_sizes) if tls_record_sizes else 0,
        'tls_record_count': len(tls_record_sizes),  # NEW: Number of TLS records
        'tls_cv': tls_cv,  # NEW: TLS record size consistency

        # Flow characteristics
        'flow_duration': flow_duration,  # NEW: Total flow duration
        'packet_rate': packet_rate,  # NEW: Packets per second
        'large_pkt_ratio': large_pkt_ratio,  # NEW: Ratio of large packets
        'small_pkt_ratio': small_pkt_ratio,  # NEW: Ratio of small packets

        # Labels
        'content_type': content_type,
        'genre': genre
    }

    return features


def process_directory(root_dir, spotify_ips):
    """
    Process all pcap files in the directory structure

    Parameters:
    - root_dir: root directory containing genre folders
    - spotify_ips: list of Spotify IP addresses

    Returns:
    - DataFrame with extracted features
    """
    all_features = []

    # Map folder names to content types
    content_type_map = {
        'podcast': 'podcast',
        'rock': 'music',
        'rap': 'music'
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
            features = extract_features(str(pcap_file), genre, content_type, spotify_ips)

            if features:
                # Extract content_id from filename
                features['content_id'] = pcap_file.stem
                all_features.append(features)

    return pd.DataFrame(all_features)


def main():
    """Main execution function"""
    print("=" * 60)
    print("Enhanced PCAP Feature Extractor for Spotify Traffic")
    print("Extracting Directional, TLS & Temporal Features")
    print("=" * 60)

    # Load Spotify IPs
    spotify_ips = load_spotify_ips()
    print(f"Loaded {len(spotify_ips)} Spotify IP(s): {', '.join(spotify_ips)}")
    print("=" * 60)

    # Process all pcap files
    df = process_directory(PCAP_ROOT, spotify_ips)

    if df.empty:
        print("\nNo features extracted. Check if:")
        print(f"1. Directory '{PCAP_ROOT}' exists")
        print("2. Subdirectories (podcast, rock, rap) contain .pcap files")
        print(f"3. Pcap files contain traffic with IPs: {spotify_ips}")
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

    # Display feature statistics by content type
    print("\n\nKey Feature Comparison (Music vs Podcast):")
    comparison_features = ['downstream_cv', 'pkt_cv', 'burst_cv', 'iat_cv',
                           'packet_rate', 'downstream_throughput', 'large_pkt_ratio']

    for feature in comparison_features:
        if feature in df.columns:
            print(f"\n{feature}:")
            print(df.groupby('content_type')[feature].agg(['mean', 'std']))

    # Display directional statistics
    print("\n\nDirectional Traffic Analysis:")
    print(f"Average upstream packets: {df['pkt_count_up'].mean():.1f}")
    print(f"Average downstream packets: {df['pkt_count_down'].mean():.1f}")
    print(f"Average bytes ratio (down/up): {df['bytes_ratio'].mean():.2f}")
    print(f"Average packet count ratio (down/up): {df['pkt_count_ratio'].mean():.2f}")


if __name__ == "__main__":
    main()