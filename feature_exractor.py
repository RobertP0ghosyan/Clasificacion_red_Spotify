#!/usr/bin/env python3
"""
PCAP Feature Extractor for Spotify Traffic Classification
Extracts directional and TLS features from pcap files
"""

import os
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, Raw
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration
SPOTIFY_IP = "35.186.224.26"
PCAP_ROOT = "pcap"
OUTPUT_FILE = "spotify_features.csv"


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


def extract_features(pcap_file, genre, content_type):
    """
    Extract directional and TLS features from pcap file

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

    # Filter packets with Spotify IP (source or destination)
    filtered_pkts = [pkt for pkt in packets if IP in pkt and
                     (pkt[IP].src == SPOTIFY_IP or pkt[IP].dst == SPOTIFY_IP)]

    if len(filtered_pkts) == 0:
        print(f"No packets found with IP {SPOTIFY_IP} in {pcap_file}")
        return None

    # Separate upstream and downstream packets
    upstream_pkts = []  # Client to Spotify
    downstream_pkts = []  # Spotify to Client
    all_pkt_sizes = []
    inter_arrivals = []
    tls_record_sizes = []

    timestamps = []

    for pkt in filtered_pkts:
        pkt_size = len(pkt)
        all_pkt_sizes.append(pkt_size)
        timestamps.append(float(pkt.time))

        # Determine direction
        if pkt[IP].dst == SPOTIFY_IP:
            # Client to server (upstream)
            upstream_pkts.append(pkt_size)
        else:
            # Server to client (downstream)
            downstream_pkts.append(pkt_size)

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
    bursts = calculate_bursts(timestamps)

    # Calculate bytes ratio
    upstream_bytes = sum(upstream_pkts)
    downstream_bytes = sum(downstream_pkts)
    bytes_ratio = downstream_bytes / upstream_bytes if upstream_bytes > 0 else 0

    # Calculate features
    features = {
        'pkt_mean_size': np.mean(all_pkt_sizes) if all_pkt_sizes else 0,
        'pkt_max_size': max(all_pkt_sizes) if all_pkt_sizes else 0,
        'pkt_count_up': len(upstream_pkts),
        'pkt_count_down': len(downstream_pkts),
        'burst_mean': np.mean(bursts) if bursts else 0,
        'burst_max': max(bursts) if bursts else 0,
        'bytes_ratio': bytes_ratio,
        'iat_std': np.std(inter_arrivals) if len(inter_arrivals) > 0 else 0,
        'tls_record_mean': np.mean(tls_record_sizes) if tls_record_sizes else 0,
        'content_type': content_type,
        'genre': genre
    }

    return features


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
            features = extract_features(str(pcap_file), genre, content_type)

            if features:
                # Extract content_id from filename
                features['content_id'] = pcap_file.stem
                all_features.append(features)

    return pd.DataFrame(all_features)


def main():
    """Main execution function"""
    print("=" * 60)
    print("PCAP Feature Extractor for Spotify Traffic")
    print("Extracting Directional & TLS Features")
    print(f"Filtering traffic for IP: {SPOTIFY_IP}")
    print("=" * 60)

    # Process all pcap files
    df = process_directory(PCAP_ROOT)

    if df.empty:
        print("\nNo features extracted. Check if:")
        print(f"1. Directory '{PCAP_ROOT}' exists")
        print("2. Subdirectories (podcast, rock, rap) contain .pcap files")
        print(f"3. Pcap files contain traffic with IP {SPOTIFY_IP}")
        return

    # Reorder columns
    column_order = ['content_id', 'pkt_mean_size', 'pkt_max_size',
                    'pkt_count_up', 'pkt_count_down', 'burst_mean',
                    'burst_max', 'bytes_ratio', 'iat_std', 'tls_record_mean',
                    'content_type', 'genre']

    df = df[column_order]

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print(f"Feature extraction complete!")
    print(f"Total files processed: {len(df)}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"{'=' * 60}\n")

    # Display summary statistics
    print("Summary by genre:")
    print(df.groupby('genre').size())

    print("\nSummary by content type:")
    print(df.groupby('content_type').size())

    print("\nFirst few rows:")
    print(df.head())

    # Display feature statistics
    print("\n\nFeature Statistics:")
    print(df.describe())

    # Display directional statistics
    print("\n\nDirectional Traffic Analysis:")
    print(f"Average upstream packets: {df['pkt_count_up'].mean():.1f}")
    print(f"Average downstream packets: {df['pkt_count_down'].mean():.1f}")
    print(f"Average bytes ratio (down/up): {df['bytes_ratio'].mean():.2f}")


if __name__ == "__main__":
    main()