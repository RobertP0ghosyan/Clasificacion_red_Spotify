#!/usr/bin/env python3
"""
Helper script to identify Spotify server IPs from network traffic.
Run this while Spotify is streaming to capture actual server IPs.
"""

from scapy.all import sniff, IP, TCP
from collections import Counter
import time

INTERFACE = "enp0s3"  # Change to your network interface
CAPTURE_DURATION = 30  # Capture for 30 seconds
OUTPUT_FILE = "spotify_ips.txt"

# Known Spotify domains for reference
SPOTIFY_DOMAINS = [
    "audio-fa.scdn.co",
    "audio4-fa.scdn.co",
    "audio-ak-spotify-com.akamaized.net",
    "heads-fa.spotify.com"
]


def capture_https_traffic():
    """Capture HTTPS traffic and identify most active IPs"""
    print("=" * 60)
    print("SPOTIFY IP DISCOVERY TOOL")
    print("=" * 60)
    print(f"\nCapturing HTTPS traffic for {CAPTURE_DURATION} seconds...")
    print(f"Interface: {INTERFACE}")
    print("\n⚠️  IMPORTANT: Start playing music on Spotify NOW!")
    print("   Make sure it's actively streaming (not paused)")

    input("\nPress ENTER when Spotify is playing...")

    print(f"\n[+] Capturing for {CAPTURE_DURATION} seconds...")
    print("    (Keep Spotify playing during this time)")

    packets = sniff(
        iface=INTERFACE,
        filter="tcp port 443",
        timeout=CAPTURE_DURATION,
        store=True
    )

    print(f"[+] Captured {len(packets)} HTTPS packets")
    return packets


def analyze_traffic(packets):
    """Analyze captured traffic to identify Spotify servers"""
    print("\n[+] Analyzing traffic patterns...")

    # Count traffic per destination IP
    ip_stats = {}

    for pkt in packets:
        if not pkt.haslayer(IP):
            continue

        dst_ip = pkt[IP].dst
        src_ip = pkt[IP].src

        # Track destination IPs (servers)
        if dst_ip not in ip_stats:
            ip_stats[dst_ip] = {
                'packets': 0,
                'bytes': 0,
                'src_ips': set()
            }

        ip_stats[dst_ip]['packets'] += 1
        ip_stats[dst_ip]['bytes'] += len(pkt)
        ip_stats[dst_ip]['src_ips'].add(src_ip)

    # Sort by packet count (streaming servers will have most traffic)
    sorted_ips = sorted(
        ip_stats.items(),
        key=lambda x: x[1]['packets'],
        reverse=True
    )

    return sorted_ips


def identify_spotify_ips(sorted_ips, top_n=10):
    """Identify likely Spotify server IPs"""
    print("\n" + "=" * 60)
    print("TOP DESTINATION IPs (Likely Servers)")
    print("=" * 60)
    print(f"\n{'Rank':<6} {'IP Address':<18} {'Packets':<10} {'Total Bytes':<15} {'Likely Spotify?'}")
    print("-" * 70)

    spotify_candidates = []

    for i, (ip, stats) in enumerate(sorted_ips[:top_n], 1):
        packets = stats['packets']
        bytes_total = stats['bytes']

        # Heuristics to identify Spotify servers:
        # 1. High packet count (> 50 packets in 30s)
        # 2. High bandwidth (> 100KB)
        # 3. Consistent traffic

        is_likely_spotify = packets > 50 and bytes_total > 100000
        likelihood = "🎵 YES" if is_likely_spotify else "   Maybe"

        print(f"{i:<6} {ip:<18} {packets:<10} {bytes_total:<15,} {likelihood}")

        if is_likely_spotify:
            spotify_candidates.append(ip)

    return spotify_candidates


def save_spotify_ips(ip_list):
    """Save identified Spotify IPs to file"""
    if not ip_list:
        print("\n⚠️  No Spotify IPs identified!")
        print("   Make sure Spotify was actively streaming during capture.")
        return False

    print(f"\n[+] Saving {len(ip_list)} Spotify IPs to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, 'w') as f:
        f.write("# Spotify Server IPs\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# These IPs were captured while Spotify was streaming\n\n")

        for ip in ip_list:
            f.write(f"{ip}\n")

    print(f"[+] Saved to {OUTPUT_FILE}")
    return True


def verify_ips():
    """Instructions to verify the captured IPs"""
    print("\n" + "=" * 60)
    print("VERIFICATION STEPS")
    print("=" * 60)
    print("\n1. Check the generated file:")
    print(f"   cat {OUTPUT_FILE}")
    print("\n2. Test with the live predictor:")
    print("   sudo python live_predictor.py")
    print("\n3. If you get 'No packets matched' errors:")
    print("   - Try running this script again")
    print("   - Or delete spotify_ips.txt to disable filtering")
    print("\n4. Optional: Look up IPs to verify they belong to Spotify:")
    print("   whois <IP_ADDRESS>")


def main():
    try:
        # Capture traffic
        packets = capture_https_traffic()

        if not packets:
            print("\n❌ No packets captured!")
            print("   Check:")
            print("   1. Network interface is correct")
            print("   2. Running with sudo privileges")
            print("   3. Spotify is streaming")
            return

        # Analyze traffic
        sorted_ips = analyze_traffic(packets)

        # Identify Spotify servers
        spotify_ips = identify_spotify_ips(sorted_ips)

        # Save to file
        if save_spotify_ips(spotify_ips):
            verify_ips()

        print("\n" + "=" * 60)
        print("✅ DONE!")
        print("=" * 60)

    except PermissionError:
        print("\n❌ Permission denied!")
        print("   Run this script with sudo:")
        print("   sudo python find_spotify_ips.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()