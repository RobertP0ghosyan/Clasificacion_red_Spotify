import numpy as np
import time
import csv
import os
from scapy.all import sniff, wrpcap
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

# ADD YOUR PLAYLISTS HERE (just the ID from the URL)
MUSIC_PLAYLISTS = [
    "7IddiFVjAJbTLniq82Vusj",  # Pink Floyd Best Of
    "5HpkkM0bOPDUgLcho7nCoZ",  # Tame Impala best songs
    "28nxGp2hLho3BA0dX3cb5P",  # THE BEST OF RADIOHEAD
    "6Fs9lBMpHdqjvQ6wCPDnKc",  # Peak Kanye
]

# ADD YOUR PODCAST PLAYLISTS HERE (just the ID from the URL)
# Using playlists ensures different episodes each time, better for ML diversity
PODCAST_PLAYLISTS = [
    "5icMx65GADu8ICFmK7BwrL",  # Top 10 podcasts for life
    "38he99wNRz1QU6mrOAeyw9",  # podcasts that changed my life <3
]

class SpotifyGenreClassificationDataset:
    def __init__(self, interface="enp0s3", tracks_per_playlist=10, episodes_per_playlist=10):
        self.spotify_client = None
        self.interface = interface
        self.dataset_dir = 'dataset_classification'  # New folder for classification
        self.pcap_dir = 'pcap_classification'  # New folder for classification
        self.current_capture = []
        self.last_packet_time = None
        self.tracks_per_playlist = tracks_per_playlist
        self.episodes_per_playlist = episodes_per_playlist

        # Will store fetched content
        self.music_tracks = []
        self.podcast_episodes = []

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.pcap_dir, exist_ok=True)

        # CSV filename with timestamp for unique identification
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.dataset_file = f"{self.dataset_dir}/spotify_classification_{timestamp}.csv"

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

        # Verify active device
        devices = self.spotify_client.devices()
        if not devices['devices']:
            raise Exception("No active Spotify device found! Please open Spotify on a device first.")
        print(f"    ✓ Active device found: {devices['devices'][0]['name']}")

    def fetch_playlist_tracks(self):
        """Fetch actual track URIs from playlists"""
        print("\n" + "=" * 70)
        print("FETCHING TRACKS FROM PLAYLISTS")
        print("=" * 70)

        for playlist_id in MUSIC_PLAYLISTS:
            try:
                playlist = self.spotify_client.playlist(playlist_id)
                print(f"\n📀 Playlist: {playlist['name']}")
                print(f"   Total tracks: {playlist['tracks']['total']}")

                results = self.spotify_client.playlist_tracks(playlist_id, limit=self.tracks_per_playlist)

                tracks_added = 0
                for item in results['items']:
                    if item['track']:
                        track = item['track']
                        self.music_tracks.append({
                            'uri': track['uri'],
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'artist_id': track['artists'][0]['id']
                        })
                        print(f"   ✓ {track['artists'][0]['name']} - {track['name']}")
                        tracks_added += 1

                print(f"   Added {tracks_added} tracks from this playlist")

            except Exception as e:
                print(f"   ✗ Error fetching playlist {playlist_id}: {e}")

        print(f"\n📊 Total music tracks collected: {len(self.music_tracks)}")

    def fetch_podcast_episodes(self):
        """Fetch actual episode URIs from podcast playlists"""
        print("\n" + "=" * 70)
        print("FETCHING EPISODES FROM PODCAST PLAYLISTS")
        print("=" * 70)

        for playlist_id in PODCAST_PLAYLISTS:
            try:
                playlist = self.spotify_client.playlist(playlist_id)
                print(f"\n🎙️  Playlist: {playlist['name']}")
                print(f"   Total episodes: {playlist['tracks']['total']}")

                results = self.spotify_client.playlist_tracks(playlist_id, limit=self.episodes_per_playlist)

                episodes_added = 0
                for item in results['items']:
                    if item['track']:
                        episode = item['track']
                        self.podcast_episodes.append({
                            'uri': episode['uri'],
                            'name': episode['name'],
                            'show': episode.get('show', {}).get('name', 'Unknown Show')
                        })
                        print(f"   ✓ {episode['name']}")
                        episodes_added += 1

                print(f"   Added {episodes_added} episodes from this playlist")

            except Exception as e:
                print(f"   ✗ Error fetching playlist {playlist_id}: {e}")

        print(f"\n📊 Total podcast episodes collected: {len(self.podcast_episodes)}")

    def packet_callback(self, packet):
        """Callback for each captured packet"""
        arrival_time = packet.time
        payload_size = len(packet)

        inter_arrival = 0
        if self.last_packet_time is not None:
            inter_arrival = arrival_time - self.last_packet_time
        self.last_packet_time = arrival_time

        self.current_capture.append({
            "arrival_time": arrival_time,
            "payload_size": payload_size,
            "inter_arrival": inter_arrival
        })

    def compute_flow_features(self):
        """Compute traffic features for current capture"""
        if not self.current_capture:
            print("    Warning: No packets captured!")
            return {
                "pkt_size_mean": 0, "pkt_size_std": 0, "pkt_size_cv": 0,
                "inter_mean": 0, "inter_std": 0, "inter_cv": 0, "p95_inter": 0,
                "burst_mean": 0, "burst_max": 0, "num_silence_gaps": 0,
                "silence_ratio": 0, "flow_duration": 0, "pkt_rate": 0
            }

        pkt_sizes = np.array([p["payload_size"] for p in self.current_capture])
        inter_arrivals = np.array([p["inter_arrival"] for p in self.current_capture[1:]])
        timestamps = np.array([p["arrival_time"] for p in self.current_capture])

        # Packet size features
        pkt_size_mean = np.mean(pkt_sizes)
        pkt_size_std = np.std(pkt_sizes)
        pkt_size_cv = pkt_size_std / pkt_size_mean if pkt_size_mean else 0

        # Inter-packet arrival features
        inter_mean = np.mean(inter_arrivals) if len(inter_arrivals) > 0 else 0
        inter_std = np.std(inter_arrivals) if len(inter_arrivals) > 0 else 0
        inter_cv = inter_std / inter_mean if inter_mean else 0
        p95_inter = np.percentile(inter_arrivals, 95) if len(inter_arrivals) > 0 else 0

        # Burst detection (packets within 0.5s)
        BURST_WINDOW = 0.5
        bursts = []
        window = []
        for t in timestamps:
            window = [x for x in window if t - x <= BURST_WINDOW]
            window.append(t)
            bursts.append(len(window))
        burst_mean = np.mean(bursts) if bursts else 0
        burst_max = max(bursts) if bursts else 0

        # Silence gaps (>2s)
        SILENCE_THRESHOLD = 2.0
        silence_gaps = [x for x in inter_arrivals if x > SILENCE_THRESHOLD]
        num_silence_gaps = len(silence_gaps)
        silence_ratio = sum(silence_gaps) / sum(inter_arrivals) if sum(inter_arrivals) > 0 else 0

        # Flow duration
        flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

        # Packet rate
        pkt_rate = len(pkt_sizes) / flow_duration if flow_duration > 0 else 0

        return {
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
            print(f"\n[{index + 1}/{total}] Capturing {content_type}")
            print(f"   {content_info['artist']} - {content_info['name']}")
        else:
            print(f"\n[{index + 1}/{total}] Capturing {content_type}")
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

        # Save PCAP
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_uri = content_info['uri'].replace(":", "_")
        pcap_filename = f"{self.pcap_dir}/{timestamp}_{safe_uri}_{self.audio_quality}.pcap"
        wrpcap(pcap_filename, packets)
        print(f"    Saved PCAP: {pcap_filename}")

        # Get genre and compute features
        if content_type == "music":
            genre = self.get_genre_for_track(content_info['artist_id'])
        else:
            genre = "podcast"

        flow_features = self.compute_flow_features()

        return {
            "content_type": content_type,
            "content_id": content_info['uri'],
            "genre": genre,
            "packets": self.current_capture,
            **flow_features
        }

    def save_dataset(self, data):
        """Save captured data to CSV - optimized for genre classification"""
        data = [d for d in data if d is not None]

        if not data:
            print("No data to save!")
            return

        file_exists = os.path.exists(self.dataset_file)

        with open(self.dataset_file, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                # CSV header: Classification columns
                # content_id: Track duplicates and identify specific songs
                # num_packets: Important feature - podcasts typically have more packets than music
                writer.writerow([
                    "content_type",  # Label 1: music or podcast
                    "genre",  # Label 2: specific genre or 'podcast'
                    "content_id",  # Track URI for duplicate detection
                    "num_packets",  # Feature: differs between music and podcasts
                    "pkt_size_mean", "pkt_size_std", "pkt_size_cv",
                    "inter_mean", "inter_std", "inter_cv", "p95_inter",
                    "burst_mean", "burst_max", "num_silence_gaps",
                    "silence_ratio", "flow_duration", "pkt_rate"
                ])

            for item in data:
                writer.writerow([
                    item["content_type"],
                    item["genre"],
                    item["content_id"],
                    len(item["packets"]),
                    item["pkt_size_mean"],
                    item["pkt_size_std"],
                    item["pkt_size_cv"],
                    item["inter_mean"],
                    item["inter_std"],
                    item["inter_cv"],
                    item["p95_inter"],
                    item["burst_mean"],
                    item["burst_max"],
                    item["num_silence_gaps"],
                    item["silence_ratio"],
                    item["flow_duration"],
                    item["pkt_rate"]
                ])

        print(f"\nDataset saved to {self.dataset_file}")

    def generate_dataset(self):
        """Main method to generate the dataset"""
        try:
            # Setup
            self.setup_spotify_client()

            # Fetch all tracks and episodes from playlists
            if MUSIC_PLAYLISTS:
                self.fetch_playlist_tracks()

            if PODCAST_PLAYLISTS:
                self.fetch_podcast_episodes()

            total_items = len(self.music_tracks) + len(self.podcast_episodes)

            if total_items == 0:
                print("\n✗ No content found! Add playlist IDs to the configuration.")
                return

            # Summary
            print("\n" + "=" * 50)
            print("COLLECTION SUMMARY")
            print("=" * 50)
            print(f"Music tracks: {len(self.music_tracks)}")
            print(f"Podcast episodes: {len(self.podcast_episodes)}")
            print(f"Total items: {total_items}")
            print(f"Estimated time: ~{total_items * (CAPTURE_DURATION + 9) // 60} minutes")
            print("\n🎯 Purpose: Genre/Content Classification")
            print("   - Binary: Music vs Podcast")
            print("   - Multi-class: Specific music genres + Podcast")
            print("=" * 50)

            input("\nMake sure Spotify is open on a device, then press Enter to start data collection...")

            # Capture data
            captured_data = []
            current_item = 0

            # Combine all content into one list
            all_content = []
            for track in self.music_tracks:
                all_content.append(("music", track))
            for episode in self.podcast_episodes:
                all_content.append(("podcast", episode))

            # Capture all content
            for content_type, content_info in all_content:
                content_data = self.capture_content_traffic(content_type, content_info, current_item, total_items)
                if content_data:
                    captured_data.append(content_data)

                current_item += 1
                if current_item < total_items:
                    print("    Waiting 5 seconds before next capture...")
                    time.sleep(5)

            # Save the dataset
            self.save_dataset(captured_data)

            # Summary with genre breakdown
            print("\n" + "=" * 50)
            print("Dataset generation complete!")
            print("=" * 50)
            print(f"Total content items captured: {len(captured_data)}/{total_items}")

            total_packets = sum(len(item["packets"]) for item in captured_data)
            print(f"Total packets captured: {total_packets}")

            # Genre distribution
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
    print("\n🎯 PURPOSE:")
    print("   Generate dataset for ML classification tasks:")
    print("   1. Binary Classification: Music vs Podcast")
    print("   2. Multi-class Classification: Specific genres + Podcast")
    print("\n⚠️  REQUIREMENTS:")
    print("1. Run this script with sudo/administrator privileges")
    print("2. Ensure Spotify credentials are set in .env file:")
    print("   - SPOTIFY_CLIENT_ID")
    print("   - SPOTIFY_CLIENT_SECRET")
    print("   - SPOTIFY_REDIRECT_URI")
    print("3. Update MUSIC_PLAYLISTS and PODCAST_PLAYLISTS at top of file")
    print("4. Install required packages:")
    print("   pip install scapy spotipy python-dotenv numpy")
    print("5. Have Spotify open and playing on a device")
    print("=" * 70 + "\n")

    # Ask how many tracks/episodes per playlist
    print("\nConfigure how many items to fetch:")
    while True:
        try:
            tracks_per = int(input("Tracks per music playlist (default 10): ").strip() or "10")
            episodes_per = int(input("Episodes per podcast playlist (default 10): ").strip() or "10")
            if tracks_per > 0 and episodes_per > 0:
                break
            print("Please enter positive numbers")
        except ValueError:
            print("Please enter valid numbers")

    generator = SpotifyGenreClassificationDataset(
        tracks_per_playlist=tracks_per,
        episodes_per_playlist=episodes_per
    )
    generator.generate_dataset()