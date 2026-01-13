import time
import os
import threading
from scapy.all import sniff, wrpcap, IP
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
IP_CACHE_FILE = "spotify_ips.json"

# Load cached Spotify IPs - REQUIRED, no fallback
try:
    import json

    with open(IP_CACHE_FILE, 'r') as f:
        cached_data = json.load(f)
        SPOTIFY_IPS = set(cached_data.get('ips', []))
        print(f"[*] Loaded {len(SPOTIFY_IPS)} cached Spotify IPs from {IP_CACHE_FILE}")
except FileNotFoundError:
    SPOTIFY_IPS = set()
    print(f"[!] ERROR: {IP_CACHE_FILE} not found!")
    print(f"[!] You MUST run discovery mode first:")
    print(f"[!]   sudo python script.py --discover")

MUSIC_PLAYLISTS = [
    # "7IddiFVjAJbTLniq82Vusj",  # Pink Floyd Best Of
    # "5HpkkM0bOPDUgLcho7nCoZ",  # Tame Impala best songs
    # "28nxGp2hLho3BA0dX3cb5P",  # THE BEST OF RADIOHEAD

    # "6Fs9lBMpHdqjvQ6wCPDnKc",  # Peak Kanye
    # "35kZMub9UFGSheeghSXBfw",  # (neo-psychedelic) Best of Tame Impala
    # "4gHuAdOjAZHMb6WYKQhbLD",  # (neo-psychedelic) mgmt need to change to indie -- tame impala alo indie
    # "3IffYurXS0a9WC3SikI4TV",  # travis(rap) best songs and hardest hits

    # ----------------------------------------------------------------
    # "0U3ACsVhROtNwwacDmhcuR",  # (Progressive Rock) 25 King Crimson
    # "4yebu47SKvUq8aWmTu1cRc",  # david bowie Art Rock

    # edm
    "1mkinKlTq2OV9MCE5Nkpp9",
    "10PXjjuLhwtYRZtJkgixLO",
    "6Sv7aZ1fHZVEWfGdhqWn87",
    "0yskWBwX31blZR9bVCBZTL",
]

# ADD YOUR PODCAST PLAYLISTS HERE (just the ID from the URL)
# Using playlists ensures different episodes each time, better for ML diversity
PODCAST_PLAYLISTS = [
    # "5icMx65GADu8ICFmK7BwrL",  # Top 10 podcasts for life
    # "38he99wNRz1QU6mrOAeyw9",  # podcasts that changed my life <3
    # "4DX89yK57dk2m5OztHqNPK",  # best true crime podcasts
    # "5lNiCLt9Rx2U3CGX2MxFcH",  # philosophy podcasts
]


def is_spotify_packet(packet):
    if not packet.haslayer(IP):
        return False

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst

    if src_ip in SPOTIFY_IPS or dst_ip in SPOTIFY_IPS:
        return True

    return False

def discover_and_save_spotify_ips(interface="enp0s3", duration=10):
    print("\n" + "=" * 60)
    print("🔍 SPOTIFY IP DISCOVERY MODE")
    print("=" * 60)
    print(f"Scanning for {duration} seconds to find Spotify server IPs.")
    print("Make sure Spotify is ACTIVELY STREAMING during this scan!")
    print("=" * 60)

    input("\n  Press ENTER to start discovery scan...")

    discovered_ips = set()
    packet_count = 0

    def discovery_callback(packet):
        nonlocal packet_count
        packet_count += 1

        if packet.haslayer(IP):
            if len(packet) > 500:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst

                if not (src_ip.startswith('192.168.') or src_ip.startswith('10.') or
                        src_ip.startswith('172.') or src_ip.startswith('127.')):
                    discovered_ips.add(src_ip)

                if not (dst_ip.startswith('192.168.') or dst_ip.startswith('10.') or
                        dst_ip.startswith('172.') or dst_ip.startswith('127.')):
                    discovered_ips.add(dst_ip)

    print(f"\n[*] Scanning for {duration} seconds...")
    sniff(iface=interface, prn=discovery_callback, timeout=duration, store=False)

    print(f"\n[+] Scan complete!")
    print(f"    Total packets: {packet_count}")
    print(f"    Potential Spotify IPs: {len(discovered_ips)}")

    if discovered_ips:
        print("\n✓ Discovered IPs:")
        for ip in sorted(discovered_ips):
            print(f"      - {ip}")

        cache_data = {
            'ips': list(discovered_ips),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'duration': duration
        }

        with open(IP_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)

        print(f"\n✓ IPs saved to {IP_CACHE_FILE}")
        return discovered_ips
    else:
        print("\n⚠️  No IPs discovered!")
        return set()


class SpotifyPcapCapture:
    def __init__(self, interface="enp0s3", tracks_per_playlist=10, episodes_per_playlist=10):
        self.spotify_client = None
        self.interface = interface
        self.pcap_dir = 'pcap'
        self.tracks_per_playlist = tracks_per_playlist
        self.episodes_per_playlist = episodes_per_playlist
        self.playback_error = None

        self.music_tracks = []
        self.podcast_episodes = []

        os.makedirs(self.pcap_dir, exist_ok=True)

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
            raise Exception("No active Spotify device found!")
        print(f"    ✓ Active device: {devices['devices'][0]['name']}")

    def fetch_playlist_tracks(self):
        """Fetch track URIs from playlists"""
        print("\n" + "=" * 70)
        print("FETCHING TRACKS FROM PLAYLISTS")
        print("=" * 70)

        for playlist_id in MUSIC_PLAYLISTS:
            try:
                playlist = self.spotify_client.playlist(playlist_id)
                print(f"\n📀 Playlist: {playlist['name']}")

                results = self.spotify_client.playlist_tracks(playlist_id, limit=self.tracks_per_playlist)

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
            except Exception as e:
                print(f"   ✗ Error: {e}")

        print(f"\n📊 Total music tracks: {len(self.music_tracks)}")

    def fetch_podcast_episodes(self):
        """Fetch episode URIs from playlists"""
        print("\n" + "=" * 70)
        print("FETCHING EPISODES FROM PODCAST PLAYLISTS")
        print("=" * 70)

        for playlist_id in PODCAST_PLAYLISTS:
            try:
                playlist = self.spotify_client.playlist(playlist_id)
                print(f"\n🎙️  Playlist: {playlist['name']}")

                results = self.spotify_client.playlist_tracks(playlist_id, limit=self.episodes_per_playlist)

                for item in results['items']:
                    if item['track']:
                        episode = item['track']
                        self.podcast_episodes.append({
                            'uri': episode['uri'],
                            'name': episode['name'],
                            'show': episode.get('show', {}).get('name', 'Unknown')
                        })
                        print(f"   ✓ {episode['name']}")
            except Exception as e:
                print(f"   ✗ Error: {e}")

        print(f"\n📊 Total podcast episodes: {len(self.podcast_episodes)}")

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

        self.playback_error = None

        def start_playback():
            time.sleep(0.5)
            try:
                self.spotify_client.start_playback(uris=[content_info['uri']], position_ms=0)
            except Exception as e:
                self.playback_error = str(e)

        playback_thread = threading.Thread(target=start_playback)
        playback_thread.daemon = True
        playback_thread.start()

        print(f"    Capturing packets for {CAPTURE_DURATION} seconds...")
        print(f"    Using {len(SPOTIFY_IPS)} IPs from {IP_CACHE_FILE}")

        # Capture ALL traffic (no filter), then filter by IP after
        packets = sniff(
            iface=self.interface,
            filter="",
            timeout=CAPTURE_DURATION,
            store=True,
        )

        if self.playback_error:
            print(f"     Playback error: {self.playback_error}")
            return None

        # Filter packets by Spotify IPs AFTER capture (ONLY cached IPs)
        spotify_packets = [p for p in packets if is_spotify_packet(p)]

        print(f"    Total packets captured: {len(packets)}")
        print(f"    Spotify packets (IP-filtered): {len(spotify_packets)}")
        if len(packets) > 0:
            print(f"    Filter efficiency: {len(spotify_packets) / len(packets) * 100:.1f}%")

        if spotify_packets:
            # Save PCAP with Spotify packets only
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_name = content_info['name'][:30].replace('/', '_').replace('\\', '_')

            # Get genre for filename
            if content_type == "music":
                genre = self.get_genre_for_track(content_info['artist_id'])
            else:
                genre = "podcast"

            pcap_filename = f"{self.pcap_dir}/{content_type}_{genre}_{timestamp}_{safe_name}.pcap"
            wrpcap(pcap_filename, spotify_packets)
            print(f"    ✓ Saved PCAP: {pcap_filename}")
            return pcap_filename
        else:
            print(f"     No Spotify packets captured!")
            return None

    def generate_dataset(self):
        try:
            # Check if we have cached IPs - REQUIRED
            if not SPOTIFY_IPS:
                print("\n" + "=" * 60)
                print(" ERROR: No Spotify IPs loaded!")
                print("=" * 60)
                print(f"You MUST run discovery mode first to create {IP_CACHE_FILE}")
                print("\nRun this command:")
                print(f"  sudo python {os.path.basename(__file__)} --discover")
                print("\nThis will:")
                print("1. Capture traffic while Spotify is streaming")
                print("2. Identify Spotify server IPs")
                print(f"3. Save them to {IP_CACHE_FILE}")
                print("=" * 60)
                return

            self.setup_spotify_client()

            if MUSIC_PLAYLISTS:
                self.fetch_playlist_tracks()

            if PODCAST_PLAYLISTS:
                self.fetch_podcast_episodes()

            total_items = len(self.music_tracks) + len(self.podcast_episodes)

            if total_items == 0:
                print("\n✗ No content found! Add playlist IDs.")
                return

            print("\n" + "=" * 60)
            print("COLLECTION SUMMARY")
            print("=" * 60)
            print(f"Music tracks: {len(self.music_tracks)}")
            print(f"Podcast episodes: {len(self.podcast_episodes)}")
            print(f"Total items: {total_items}")
            print(f"Estimated time: ~{total_items * (CAPTURE_DURATION + 9) // 60} minutes")
            print(f"\n✓ Using {len(SPOTIFY_IPS)} cached Spotify IPs from {IP_CACHE_FILE}")
            print("=" * 60)

            input("\nMake sure Spotify is open, then press Enter...")

            print("\n" + "=" * 60)
            print("STARTING CAPTURE")
            print("=" * 60)

            captured_count = 0
            current_item = 0

            # Capture music tracks
            for track in self.music_tracks:
                result = self.capture_content_traffic("music", track, current_item, total_items)
                if result:
                    captured_count += 1

                current_item += 1
                if current_item < total_items:
                    print("    Waiting 5 seconds before next capture...")
                    time.sleep(5)

            # Capture podcast episodes
            for episode in self.podcast_episodes:
                result = self.capture_content_traffic("podcast", episode, current_item, total_items)
                if result:
                    captured_count += 1

                current_item += 1
                if current_item < total_items:
                    print("    Waiting 5 seconds before next capture...")
                    time.sleep(5)

            # Summary
            print("\n" + "=" * 60)
            print("CAPTURE COMPLETE")
            print("=" * 60)
            print(f"Successfully captured: {captured_count}/{total_items} items")
            print(f"PCAPs saved to: {self.pcap_dir}/")
            print("=" * 60)

        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys

    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--discover":
            interface = input("Network interface (default: enp0s3): ").strip() or "enp0s3"
            discover_and_save_spotify_ips(interface=interface, duration=10)
        else:
            print("=" * 70)
            print("SPOTIFY PCAP CAPTURE - USES ONLY DISCOVERED IPs")
            print("=" * 70)
            print("\n📋 REQUIREMENTS:")
            print("1. sudo privileges")
            print("2. Spotify credentials in .env")
            print("3. Update MUSIC_PLAYLISTS and PODCAST_PLAYLISTS")
            print("4. pip install scapy spotipy python-dotenv")
            print("\n FIRST TIME SETUP:")
            print("   1. Run discovery mode:")
            print(f"      sudo python {os.path.basename(__file__)} --discover")
            print("   2. Start Spotify and play music during the scan")
            print("   3. Then run normal capture mode")

            if SPOTIFY_IPS:
                print(f"\n✓ Found {len(SPOTIFY_IPS)} IPs in {IP_CACHE_FILE}")
            else:
                print(f"\n No IPs found in {IP_CACHE_FILE}")
                print("   You MUST run --discover mode first!")
            print("=" * 70 + "\n")

            if not SPOTIFY_IPS:
                print("Cannot proceed without discovered IPs. Exiting...")
                exit(1)

            print("\nConfigure capture:")
            while True:
                try:
                    tracks = int(input("Tracks per playlist (default 10): ").strip() or "10")
                    episodes = int(input("Episodes per playlist (default 10): ").strip() or "10")
                    if tracks > 0 and episodes > 0:
                        break
                    print("Enter positive numbers")
                except ValueError:
                    print("Enter valid numbers")

            generator = SpotifyPcapCapture(
                tracks_per_playlist=tracks,
                episodes_per_playlist=episodes
            )
            generator.generate_dataset()

    except KeyboardInterrupt:
        print("\n\n  Interrupted")
        exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)