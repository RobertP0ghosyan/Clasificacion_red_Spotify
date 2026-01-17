#!/usr/bin/env python3
import os
import csv
import numpy as np
from collections import defaultdict
from scapy.all import rdpcap, TCP, IP

PCAP_DIR = "validated_pcaps_predict"
OUTPUT_CSV = "features_v2FINALPREDICT.csv"

# =========================
# Helpers
# =========================

def load_tcp_packets(path):
    pkts = rdpcap(path)
    return [p for p in pkts if p.haslayer(TCP) and p.haslayer(IP)]

def infer_direction(pkts):
    ip_bytes = {}
    for p in pkts:
        ip = p[IP].src
        ip_bytes[ip] = ip_bytes.get(ip, 0) + len(p)

    server_ip = max(ip_bytes, key=ip_bytes.get)
    down = [p for p in pkts if p[IP].src == server_ip]
    up   = [p for p in pkts if p[IP].src != server_ip]

    return down, up

# =========================
# Feature computation
# =========================

def compute_global_features(pkts, down, up):
    times = [p.time for p in pkts]
    duration = max(times) - min(times)

    pkt_sizes_down = [len(p) for p in down]

    return {
        "duration": duration,
        "pkts_total": len(pkts),
        "pkts_down": len(down),
        "pkts_up": len(up),
        "bytes_total": sum(len(p) for p in pkts),
        "bytes_down": sum(len(p) for p in down),
        "bytes_up": sum(len(p) for p in up),
        "bytes_ratio_down_up": (
            sum(len(p) for p in down) / sum(len(p) for p in up)
            if sum(len(p) for p in up) > 0 else 0
        ),
        "mean_pkt_size_down": np.mean(pkt_sizes_down) if pkt_sizes_down else 0,
        "std_pkt_size_down": np.std(pkt_sizes_down) if pkt_sizes_down else 0,
    }

def compute_window_features(down, t0):
    windows = {
        "0_5":   (0, 5),
        "5_15":  (5, 15),
        "15_30": (15, 30),
        "30_60": (30, 60),
    }

    feats = {}

    for name, (a, b) in windows.items():
        pkts_w = [p for p in down if a <= (p.time - t0) < b]
        bytes_w = sum(len(p) for p in pkts_w)
        dur = b - a

        feats[f"bytes_down_{name}"] = bytes_w
        feats[f"pkts_down_{name}"] = len(pkts_w)
        feats[f"bps_down_{name}"] = bytes_w / dur if dur > 0 else 0

    return feats

def compute_burstiness(down, t0):
    bins = defaultdict(int)

    for p in down:
        sec = int(p.time - t0)
        bins[sec] += len(p)

    values = np.array(list(bins.values()))
    if len(values) == 0:
        return {
            "bytes_per_sec_mean": 0,
            "bytes_per_sec_std": 0,
            "bytes_per_sec_max": 0,
            "burstiness_index": 0
        }

    mean = values.mean()
    std = values.std()

    return {
        "bytes_per_sec_mean": mean,
        "bytes_per_sec_std": std,
        "bytes_per_sec_max": values.max(),
        "burstiness_index": std / mean if mean > 0 else 0
    }

# =========================
# Main
# =========================

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    header_written = False

    for fname in sorted(os.listdir(PCAP_DIR)):
        if not fname.endswith(".pcap"):
            continue

        try:
            pkts = load_tcp_packets(os.path.join(PCAP_DIR, fname))
            if not pkts:
                continue

            down, up = infer_direction(pkts)
            if not down:
                continue

            t0 = min(p.time for p in pkts)

            features = {}
            features.update(compute_global_features(pkts, down, up))
            features.update(compute_window_features(down, t0))
            features.update(compute_burstiness(down, t0))

            if not header_written:
                writer.writerow(["pcap"] + list(features.keys()))
                header_written = True

            writer.writerow(
                [fname] + [round(v, 6) for v in features.values()]
            )

        except Exception as e:
            print(f"[!] Skipped {fname}: {e}")
