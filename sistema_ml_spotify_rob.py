import numpy as np
import time
import csv
import os
import json
from scapy.all import sniff, wrpcap, IP, TCP
from collections import defaultdict
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# Load .env file from current directory
load_dotenv()

# Configuration
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
CAPTURE_DURATION = 60  # seconds

# Config file path
CONFIG_FILE = "dataset_config.json"

class SpotifyGenreClassificationDataset:
    def __init__(self, interface="ens33", quality="high"):
        self.spotify_client = None
        self.interface = interface
        self.dataset_dir = 'dataset'
        self.pcap_dir = 'pcap_classification'
        self.current_capture = []
        self.last_packet_time = None
        self.quality = quality

        self.music_tracks = []
        self.podcast_episodes = []

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.pcap_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.dataset_file = f"{self.dataset_dir}/spotify_classification_{quality}_{timestamp}.csv"

    def load_config(self):
        """Load tracks and episodes from JSON config file"""
        print("\n" + "=" * 70)
        print("LOADING CONTENT FROM CONFIG FILE")
        print("=" * 70)
        
        if not os.path.exists(CONFIG_FILE):
            raise Exception(f"Config file '{CONFIG_FILE}' not found!")
        
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        # Load music tracks
        if "music" in config:
            print(f"\n📀 Loading {len(config['music'])} music tracks...")
            for uri in config['music']:
                try:
                    # Extract track ID from URI
                    track_id = uri.split(':')[-1]
                    track = self.spotify_client.track(track_id)
                    
                    self.music_tracks.append({
                        'uri': uri,
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'artist_id': track['artists'][0]['id']
                    })
                    print(f"   ✓ {track['artists'][0]['name']} - {track['name']}")
                except Exception as e:
                    print(f"   ✗ Error loading track {uri}: {e}")
        
        # Load podcast episodes
        if "podcast" in config:
            print(f"\n🎙️  Loading {len(config['podcast'])} podcast episodes...")
            for uri in config['podcast']:
                try:
                    # Extract episode ID from URI
                    episode_id = uri.split(':')[-1]
                    episode = self.spotify_client.episode(episode_id)
                    
                    self.podcast_episodes.append({
                        'uri': uri,
                        'name': episode['name'],
                        'show': episode.get('show', {}).get('name', 'Unknown Show')
                    })
                    print(f"   ✓ {episode['name']}")
                except Exception as e:
                    print(f"   ✗ Error loading episode {uri}: {e}")
        
        print(f"\n📊 Total music tracks loaded: {len(self.music_tracks)}")
        print(f"📊 Total podcast episodes loaded: {len(self.podcast_episodes)}")

    def setup_spotify_client(self):
        """Initialize Spotipy client"""
        print("Setting up Spotify client...")
        scope = "user-modify-playback-state user-read-playback-state"
        self.spotify_client = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=SPOTIFY_REDIRECT_URI,
            scope=scope
        ))
        print("    ✓ Spotify client authenticated")

        devices = self.spotify_client.devices()
        if not devices['devices']:
            raise Exception("No active Spotify device found! Please open Spotify on a device first.")
        print(f"    ✓ Active device found: {devices['devices'][0]['name']}")

    def packet_callback(self, packet):
        """Callback for each captured packet - ENHANCED"""
        arrival_time = packet.time
        
        # Basic packet info
        payload_size = len(packet)
        
        # TCP/IP specific features
        tcp_flags = 0
        ip_ttl = 0
        tcp_window = 0
        ip_len = 0
        
        if IP in packet:
            ip_ttl = packet[IP].ttl
            ip_len = packet[IP].len
            
        if TCP in packet:
            # TCP flags as bitmap
            tcp_flags = packet[TCP].flags
            tcp_window = packet[TCP].window

        inter_arrival = 0
        if self.last_packet_time is not None:
            inter_arrival = arrival_time - self.last_packet_time
        self.last_packet_time = arrival_time

        self.current_capture.append({
            "arrival_time": arrival_time,
            "payload_size": payload_size,
            "inter_arrival": inter_arrival,
            "tcp_flags": int(tcp_flags),
            "ip_ttl": ip_ttl,
            "tcp_window": tcp_window,
            "ip_len": ip_len
        })

    def compute_flow_features(self):
        """Compute COMPREHENSIVE traffic features for neural network training"""
        if not self.current_capture:
            print("    Warning: No packets captured!")
            return self._empty_features()

        # Extract arrays
        pkt_sizes = np.array([p["payload_size"] for p in self.current_capture])
        inter_arrivals = np.array([p["inter_arrival"] for p in self.current_capture[1:]])
        timestamps = np.array([p["arrival_time"] for p in self.current_capture])
        tcp_flags = np.array([p["tcp_flags"] for p in self.current_capture])
        ip_ttls = np.array([p["ip_ttl"] for p in self.current_capture])
        tcp_windows = np.array([p["tcp_window"] for p in self.current_capture])
        ip_lens = np.array([p["ip_len"] for p in self.current_capture])

        features = {}
        
        # ==================== PACKET SIZE FEATURES ====================
        features["pkt_size_mean"] = np.mean(pkt_sizes)
        features["pkt_size_std"] = np.std(pkt_sizes)
        features["pkt_size_min"] = np.min(pkt_sizes)
        features["pkt_size_max"] = np.max(pkt_sizes)
        features["pkt_size_median"] = np.median(pkt_sizes)
        features["pkt_size_cv"] = features["pkt_size_std"] / features["pkt_size_mean"] if features["pkt_size_mean"] else 0
        features["pkt_size_p25"] = np.percentile(pkt_sizes, 25)
        features["pkt_size_p75"] = np.percentile(pkt_sizes, 75)
        features["pkt_size_p95"] = np.percentile(pkt_sizes, 95)
        
        # Packet size distribution (histogram bins)
        hist, _ = np.histogram(pkt_sizes, bins=5, range=(0, 2000))
        for i, count in enumerate(hist):
            features[f"pkt_size_bin_{i}"] = count

        # ==================== INTER-ARRIVAL TIME FEATURES ====================
        if len(inter_arrivals) > 0:
            features["inter_mean"] = np.mean(inter_arrivals)
            features["inter_std"] = np.std(inter_arrivals)
            features["inter_min"] = np.min(inter_arrivals)
            features["inter_max"] = np.max(inter_arrivals)
            features["inter_median"] = np.median(inter_arrivals)
            features["inter_cv"] = features["inter_std"] / features["inter_mean"] if features["inter_mean"] else 0
            features["inter_p25"] = np.percentile(inter_arrivals, 25)
            features["inter_p75"] = np.percentile(inter_arrivals, 75)
            features["inter_p95"] = np.percentile(inter_arrivals, 95)
        else:
            features["inter_mean"] = 0
            features["inter_std"] = 0
            features["inter_min"] = 0
            features["inter_max"] = 0
            features["inter_median"] = 0
            features["inter_cv"] = 0
            features["inter_p25"] = 0
            features["inter_p75"] = 0
            features["inter_p95"] = 0

        # ==================== BURST FEATURES ====================
        BURST_WINDOW = 0.5
        bursts = []
        window = []
        for t in timestamps:
            window = [x for x in window if t - x <= BURST_WINDOW]
            window.append(t)
            bursts.append(len(window))
        
        features["burst_mean"] = np.mean(bursts) if bursts else 0
        features["burst_max"] = max(bursts) if bursts else 0
        features["burst_std"] = np.std(bursts) if bursts else 0
        features["burst_median"] = np.median(bursts) if bursts else 0

        # ==================== SILENCE/GAP FEATURES ====================
        SILENCE_THRESHOLD = 2.0
        silence_gaps = [x for x in inter_arrivals if x > SILENCE_THRESHOLD]
        features["num_silence_gaps"] = len(silence_gaps)
        features["silence_ratio"] = sum(silence_gaps) / sum(inter_arrivals) if sum(inter_arrivals) > 0 else 0
        features["avg_silence_duration"] = np.mean(silence_gaps) if silence_gaps else 0

        # ==================== FLOW DURATION & RATE FEATURES ====================
        flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        features["flow_duration"] = flow_duration
        features["pkt_rate"] = len(pkt_sizes) / flow_duration if flow_duration > 0 else 0
        features["byte_rate"] = np.sum(pkt_sizes) / flow_duration if flow_duration > 0 else 0
        
        # ==================== DIRECTIONALITY FEATURES ====================
        # Approximation: small packets likely ACKs (upstream), large packets data (downstream)
        small_pkts = np.sum(pkt_sizes < 100)
        large_pkts = np.sum(pkt_sizes >= 100)
        features["small_pkt_ratio"] = small_pkts / len(pkt_sizes) if len(pkt_sizes) > 0 else 0
        features["large_pkt_ratio"] = large_pkts / len(pkt_sizes) if len(pkt_sizes) > 0 else 0

        # ==================== TCP FLAGS FEATURES ====================
        # Count different flag combinations
        features["tcp_syn_count"] = np.sum((tcp_flags & 0x02) != 0)  # SYN
        features["tcp_ack_count"] = np.sum((tcp_flags & 0x10) != 0)  # ACK
        features["tcp_psh_count"] = np.sum((tcp_flags & 0x08) != 0)  # PSH
        features["tcp_fin_count"] = np.sum((tcp_flags & 0x01) != 0)  # FIN
        features["tcp_rst_count"] = np.sum((tcp_flags & 0x04) != 0)  # RST

        # ==================== IP/TCP HEADER FEATURES ====================
        features["ip_ttl_mean"] = np.mean(ip_ttls) if len(ip_ttls) > 0 else 0
        features["ip_ttl_std"] = np.std(ip_ttls) if len(ip_ttls) > 0 else 0
        features["tcp_window_mean"] = np.mean(tcp_windows) if len(tcp_windows) > 0 else 0
        features["tcp_window_std"] = np.std(tcp_windows) if len(tcp_windows) > 0 else 0
        features["ip_len_mean"] = np.mean(ip_lens) if len(ip_lens) > 0 else 0

        # ==================== TEMPORAL PATTERNS ====================
        # Divide flow into 4 quarters and analyze
        quarter_size = len(pkt_sizes) // 4
        if quarter_size > 0:
            for i in range(4):
                start = i * quarter_size
                end = start + quarter_size if i < 3 else len(pkt_sizes)
                quarter_pkts = pkt_sizes[start:end]
                features[f"quarter_{i}_mean_size"] = np.mean(quarter_pkts)
                features[f"quarter_{i}_pkt_count"] = len(quarter_pkts)
        else:
            for i in range(4):
                features[f"quarter_{i}_mean_size"] = 0
                features[f"quarter_{i}_pkt_count"] = 0

        # ==================== ADDITIONAL STATISTICAL FEATURES ====================
        features["num_packets"] = len(pkt_sizes)
        features["total_bytes"] = np.sum(pkt_sizes)
        
        # Skewness and kurtosis for packet sizes (shape of distribution)
        from scipy import stats
        features["pkt_size_skewness"] = stats.skew(pkt_sizes) if len(pkt_sizes) > 1 else 0
        features["pkt_size_kurtosis"] = stats.kurtosis(pkt_sizes) if len(pkt_sizes) > 1 else 0
        
        if len(inter_arrivals) > 1:
            features["inter_skewness"] = stats.skew(inter_arrivals)
            features["inter_kurtosis"] = stats.kurtosis(inter_arrivals)
        else:
            features["inter_skewness"] = 0
            features["inter_kurtosis"] = 0

        return features

    def _empty_features(self):
        """Return empty feature dict when no packets captured"""
        empty = {
            "pkt_size_mean": 0, "pkt_size_std": 0, "pkt_size_min": 0, "pkt_size_max": 0,
            "pkt_size_median": 0, "pkt_size_cv": 0, "pkt_size_p25": 0, "pkt_size_p75": 0, "pkt_size_p95": 0,
            "inter_mean": 0, "inter_std": 0, "inter_min": 0, "inter_max": 0, "inter_median": 0,
            "inter_cv": 0, "inter_p25": 0, "inter_p75": 0, "inter_p95": 0,
            "burst_mean": 0, "burst_max": 0, "burst_std": 0, "burst_median": 0,
            "num_silence_gaps": 0, "silence_ratio": 0, "avg_silence_duration": 0,
            "flow_duration": 0, "pkt_rate": 0, "byte_rate": 0,
            "small_pkt_ratio": 0, "large_pkt_ratio": 0,
            "tcp_syn_count": 0, "tcp_ack_count": 0, "tcp_psh_count": 0, "tcp_fin_count": 0, "tcp_rst_count": 0,
            "ip_ttl_mean": 0, "ip_ttl_std": 0, "tcp_window_mean": 0, "tcp_window_std": 0, "ip_len_mean": 0,
            "num_packets": 0, "total_bytes": 0,
            "pkt_size_skewness": 0, "pkt_size_kurtosis": 0, "inter_skewness": 0, "inter_kurtosis": 0
        }
        for i in range(5):
            empty[f"pkt_size_bin_{i}"] = 0
        for i in range(4):
            empty[f"quarter_{i}_mean_size"] = 0
            empty[f"quarter_{i}_pkt_count"] = 0
        return empty

    def get_genre_for_track(self, artist_id):
        """Get genre from artist"""
        try:
            artist = self.spotify_client.artist(artist_id)
            return artist["genres"][0] if artist["genres"] else "unknown"
        except Exception as e:
            print(f"    Warning: Could not fetch genre - {e}")
            return "unknown"

    def capture_content_traffic(self, content_type, content_info, index, total):
        """Capture traffic for a track or episode"""
        if content_type == "music":
            print(f"\n[{index + 1}/{total}] Capturing {content_type} @ {self.quality} quality")
            print(f"   {content_info['artist']} - {content_info['name']}")
        else:
            print(f"\n[{index + 1}/{total}] Capturing {content_type} @ {self.quality} quality")
            print(f"   {content_info['show']} - {content_info['name']}")

        try:
            self.spotify_client.start_playback(uris=[content_info['uri']], position_ms=0)
            time.sleep(4)
        except Exception as e:
            print(f"    Playback error: {e}")
            return None

        self.current_capture = []
        self.last_packet_time = None

        print(f"    Capturing packets for {CAPTURE_DURATION} seconds...")
        packets = sniff(
            iface=self.interface,
            filter="tcp port 443",
            prn=self.packet_callback,
            timeout=CAPTURE_DURATION,
            store=True,
        )

        print(f"    Captured {len(packets)} packets")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_uri = content_info['uri'].replace(":", "_")
        pcap_filename = f"{self.pcap_dir}/{timestamp}_{self.quality}_{safe_uri}.pcap"
        wrpcap(pcap_filename, packets)
        print(f"    Saved PCAP: {pcap_filename}")

        if content_type == "music":
            genre = self.get_genre_for_track(content_info['artist_id'])
        else:
            genre = "podcast"

        flow_features = self.compute_flow_features()

        return {
            "content_type": content_type,
            "content_id": content_info['uri'],
            "genre": genre,
            "quality": self.quality,
            **flow_features
        }

    def save_dataset(self, data):
        """Save captured data to CSV - ALL FEATURES"""
        data = [d for d in data if d is not None]

        if not data:
            print("No data to save!")
            return

        file_exists = os.path.exists(self.dataset_file)

        with open(self.dataset_file, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                # Get all feature names from first data item
                feature_keys = [k for k in data[0].keys() if k not in ['content_type', 'genre', 'quality', 'content_id']]
                
                # Header: Labels first, then all features
                header = ["content_type", "genre", "quality", "content_id"] + feature_keys
                writer.writerow(header)

            for item in data:
                feature_keys = [k for k in item.keys() if k not in ['content_type', 'genre', 'quality', 'content_id']]
                row = [
                    item["content_type"],
                    item["genre"],
                    item["quality"],
                    item["content_id"]
                ] + [item[k] for k in feature_keys]
                writer.writerow(row)

        print(f"\nDataset saved to {self.dataset_file}")
        print(f"Total features per sample: {len(feature_keys)}")

    def generate_dataset(self):
        """Main method to generate the dataset"""
        try:
            self.setup_spotify_client()

            # Load content from JSON config
            self.load_config()

            total_items = len(self.music_tracks) + len(self.podcast_episodes)

            if total_items == 0:
                print("\n✗ No content found in config file!")
                return

            print("\n" + "=" * 50)
            print("COLLECTION SUMMARY")
            print("=" * 50)
            print(f"Quality setting: {self.quality.upper()}")
            print(f"Music tracks: {len(self.music_tracks)}")
            print(f"Podcast episodes: {len(self.podcast_episodes)}")
            print(f"Total items: {total_items}")
            print(f"Estimated time: ~{total_items * (CAPTURE_DURATION + 9) // 60} minutes")
            print("\n🎯 Purpose: Genre/Content Classification with Quality Detection")
            print("   - Binary: Music vs Podcast")
            print("   - Multi-class: Specific music genres + Podcast")
            print("   - Quality detection across all classes")
            print("=" * 50)

            input("\nMake sure Spotify is open on a device, then press Enter to start data collection...")

            captured_data = []
            current_item = 0

            all_content = []
            for track in self.music_tracks:
                all_content.append(("music", track))
            for episode in self.podcast_episodes:
                all_content.append(("podcast", episode))

            for content_type, content_info in all_content:
                content_data = self.capture_content_traffic(content_type, content_info, current_item, total_items)
                if content_data:
                    captured_data.append(content_data)

                current_item += 1
                if current_item < total_items:
                    print("    Waiting 5 seconds before next capture...")
                    time.sleep(5)

            self.save_dataset(captured_data)

            print("\n" + "=" * 50)
            print("Dataset generation complete!")
            print("=" * 50)
            print(f"Quality: {self.quality.upper()}")
            print(f"Total content items captured: {len(captured_data)}/{total_items}")

            print("\n📊 Genre Distribution:")
            genre_counts = {}
            for item in captured_data:
                genre = item['genre']
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

            for genre, count in sorted(genre_counts.items()):
                print(f"   {genre}: {count} samples")

            print("\n✓ Dataset saved to: " + self.dataset_file)
            print("✓ PCAP files saved to: " + self.pcap_dir)
            print("=" * 50)

        except KeyboardInterrupt:
            print("\n\nCapture interrupted by user")
            print("Saving partial dataset...")
            if captured_data:
                self.save_dataset(captured_data)
        except Exception as e:
            print(f"\n\nError during capture: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    print("=" * 70)
    print("SPOTIFY GENRE/CONTENT CLASSIFICATION DATASET GENERATOR")
    print("=" * 70)
    print("\n  REQUIREMENTS:")
    print("1. Run this script with sudo/administrator privileges")
    print("2. Ensure Spotify credentials are set in .env file:")
    print("   - SPOTIFY_CLIENT_ID")
    print("   - SPOTIFY_CLIENT_SECRET")
    print("   - SPOTIFY_REDIRECT_URI")
    print("3. Create 'dataset_config.json' with track/episode URIs")
    print("4. Install required packages:")
    print("   pip install scapy spotipy python-dotenv numpy scipy")
    print("5. Have Spotify open and playing on a device")
    print("=" * 70 + "\n")

    # Check if config file exists
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ ERROR: Config file '{CONFIG_FILE}' not found!")
        print("\nCreate a file named 'dataset_config.json' with this structure:")
        print("""{
  "music": [
    "spotify:track:TRACK_ID_1",
    "spotify:track:TRACK_ID_2"
  ],
  "podcast": [
    "spotify:episode:EPISODE_ID_1",
    "spotify:episode:EPISODE_ID_2"
  ]
}""")
        exit(1)

    # Ask for quality setting
    print("\n🎵 Configure Spotify Quality:")
    print("   Available options: low, normal, high, very_high")
    print("   Note: Make sure this matches your Spotify app quality settings!")
    while True:
        quality_input = input("Enter quality setting (default 'high'): ").strip().lower() or "high"
        if quality_input in ["low", "normal", "high", "very_high"]:
            quality = quality_input
            break
        print("❌ Invalid quality. Choose: low, normal, high, or very_high")

    generator = SpotifyGenreClassificationDataset(quality=quality)
    generator.generate_dataset()
