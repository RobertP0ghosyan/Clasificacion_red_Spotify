#!/usr/bin/env python3
import os
import csv
import re
from scapy.all import rdpcap, TCP, IP

PCAP_DIR = "pcap_folder"
OUTPUT_CSV = "pcap_validation_predict.csv"

# =========================
# Expected contents
# =========================

VALID_IDS = {
    "2pbSCYzxrG0wa6qcj8IyiE",
    "1zMH7Nak15QthI2Y7WrCwg",
    "3eE2slIDl03bU6VYvnXQZv",
    "6YkJqm4DRjYWDfa7SyO6eq",
    "1NZibQdCjc6IqINITrjWSq",
    "5D1uq9kLKjDzVM3cDjzI1X",
}

MIN_DURATION = 10          # seconds (only to remove broken captures)
MIN_DOWN_UP_RATIO = 2.0    # streaming heuristic

# =========================
# Helpers
# =========================

def extract_content_id(filename):
    """
    Extract Spotify track/episode ID from filename.
    """
    m = re.search(r"spotify_(track|episode)_([a-zA-Z0-9]+)", filename)
    if not m:
        return None
    return m.group(2)

def analyze_pcap(path):
    try:
        pkts = rdpcap(path)
    except Exception:
        return {"status": "read_error"}

    tcp_pkts = [p for p in pkts if p.haslayer(TCP) and p.haslayer(IP)]
    if not tcp_pkts:
        return {"status": "no_tcp"}

    times = [p.time for p in tcp_pkts]
    duration = max(times) - min(times)

    # infer server by byte dominance
    ip_bytes = {}
    for p in tcp_pkts:
        ip = p[IP].src
        ip_bytes[ip] = ip_bytes.get(ip, 0) + len(p)

    server_ip = max(ip_bytes, key=ip_bytes.get)

    bytes_down = sum(len(p) for p in tcp_pkts if p[IP].src == server_ip)
    bytes_up   = sum(len(p) for p in tcp_pkts if p[IP].src != server_ip)

    ratio = bytes_down / bytes_up if bytes_up > 0 else 0

    if duration < MIN_DURATION:
        status = "too_short"
    elif ratio < MIN_DOWN_UP_RATIO:
        status = "no_downlink_dominance"
    else:
        status = "ok"

    return {
        "status": status,
        "duration_s": round(duration, 2),
        "bytes_down": bytes_down,
        "bytes_up": bytes_up,
        "bytes_total": bytes_down + bytes_up,
        "down_up_ratio": round(ratio, 3),
    }

# =========================
# Main
# =========================

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "pcap",
        "content_id",
        "content_ok",
        "status",
        "duration_s",
        "bytes_down",
        "bytes_up",
        "bytes_total",
        "down_up_ratio",
    ])

    for fname in sorted(os.listdir(PCAP_DIR)):
        if not fname.endswith(".pcap"):
            continue

        content_id = extract_content_id(fname)
        content_ok = content_id in VALID_IDS

        stats = analyze_pcap(os.path.join(PCAP_DIR, fname))

        writer.writerow([
            fname,
            content_id,
            content_ok,
            stats.get("status", ""),
            stats.get("duration_s", ""),
            stats.get("bytes_down", ""),
            stats.get("bytes_up", ""),
            stats.get("bytes_total", ""),
            stats.get("down_up_ratio", ""),
        ])
