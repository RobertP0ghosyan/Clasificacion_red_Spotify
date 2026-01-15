import time
import os
import threading
from scapy.all import sniff, wrpcap
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
CAPTURE_DURATION = 60  # seconds

MUSIC_PLAYLISTS = [
    "5HpkkM0bOPDUgLcho7nCoZ",  # Tame Impala best songs
    "7IddiFVjAJbTLniq82Vusj",  # Pink Floyd Best Of
    "28nxGp2hLho3BA0dX3cb5P",  # THE BEST OF RADIOHEAD

    # "35kZMub9UFGSheeghSXBfw",  # (neo-psychedelic) Best of Tame Impala
    # "4gHuAdOjAZHMb6WYKQhbLD",  # (neo-psychedelic) mgmt need to change to indie -- tame impala alo indie

    "6Fs9lBMpHdqjvQ6wCPDnKc",  # Peak Kanye
    "2Zz5YsBYmvWFweK5S0sRdT",  # Kendriick
    "3IffYurXS0a9WC3SikI4TV",  # travis(rap) best songs and hardest hits

    # ----------------------------------------------------------------
    # "0U3ACsVhROtNwwacDmhcuR",  # (Progressive Rock) 25 King Crimson
    # "4yebu47SKvUq8aWmTu1cRc",  # david bowie Art Rock

    # edm
    # "1mkinKlTq2OV9MCE5Nkpp9",
    # "10PXjjuLhwtYRZtJkgixLO",
    # "6Sv7aZ1fHZVEWfGdhqWn87",
    # "0yskWBwX31blZR9bVCBZTL",
]

PODCAST_PLAYLISTS = [
    # "5icMx65GADu8ICFmK7BwrL",  # Top 10 podcasts for life
    # "38he99wNRz1QU6mrOAeyw9",  # podcasts that changed my life <3
    # "4DX89yK57dk2m5OztHqNPK",  # best true crime podcasts
    # "5lNiCLt9Rx2U3CGX2MxFcH",  # philosophy podcasts
    # "316J2qxLUujvp9IGiZJ7PW"
]


class SpotifyPcapCapture:
    def __init__(self, interface="enp0s3", tracks_per_playlist=10, episodes_per_playlist=10):
        self.spotify_client = None
        self.interface = interface
        self.pcap_dir = 'pcap'
        self.tracks_per_playlist = tracks_per_playlist
        self.episodes_per_playlist = episodes_per_playlist

        self.music_tracks = []
        self.podcast_episodes = []

        os.makedirs(self.pcap_dir, exist_ok=True)

    def setup_spotify_client(self):
        """Initialize Spotipy client - connects to already running Spotify"""
        print("\nConnecting to Spotify...")
        scope = "user-modify-playback-state user-read-playback-state"

        # Get the actual user's home directory (not root's)
        import pwd
        cache_path = '.cache-spotify-pcap-token'

        self.spotify_client = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=SPOTIFY_REDIRECT_URI,
            scope=scope,
            cache_path=cache_path,
            open_browser=False  # Don't try to open browser as root
        ))
        print("    ✓ Spotify API authenticated")

        # Check for active devices
        devices = self.spotify_client.devices()
        if not devices['devices']:
            print("\n" + "=" * 70)
            print("❌ ERROR: No active Spotify device found!")
            print("=" * 70)
            print("Please make sure:")
            print("  1. Spotify is open and running on this computer or your phone")
            print("  2. You're logged in to Spotify")
            print("  3. The device is active (play something briefly to activate it)")
            print("=" * 70)
            raise Exception("No active Spotify device found!")

        active_device = devices['devices'][0]
        print(f"    ✓ Active device: {active_device['name']} ({active_device['type']})")

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
        """Capture traffic using Scapy while content plays"""
        if content_type == "music":
            print(f"\n[{index + 1}/{total}] Capturing {content_type}")
            print(f"   🎵 {content_info['artist']} - {content_info['name']}")
        else:
            print(f"\n[{index + 1}/{total}] Capturing {content_type}")
            print(f"   🎙️  {content_info['show']} - {content_info['name']}")

        # Prepare filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = content_info['name'][:30].replace('/', '_').replace('\\', '_').replace(':', '_')

        # Get genre for directory organization
        if content_type == "music":
            genre = self.get_genre_for_track(content_info['artist_id'])
        else:
            genre = "podcast"

        # Create genre-based directory structure
        genre_dir = os.path.join(self.pcap_dir, genre)
        os.makedirs(genre_dir, exist_ok=True)

        pcap_filename = os.path.join(genre_dir, f"{timestamp}_{safe_name}.pcap")

        # Start Spotify playback in a separate thread
        playback_started = threading.Event()
        playback_error = {'error': None}

        def start_playback():
            try:
                time.sleep(1)  # Small delay to ensure capture is ready
                self.spotify_client.start_playback(uris=[content_info['uri']], position_ms=0)
                playback_started.set()
            except Exception as e:
                playback_error['error'] = str(e)
                playback_started.set()

        playback_thread = threading.Thread(target=start_playback)
        playback_thread.daemon = True

        print(f"    📡 Starting packet capture for {CAPTURE_DURATION} seconds...")
        playback_thread.start()

        # Capture packets with Scapy
        try:
            packets = sniff(
                iface=self.interface,
                timeout=CAPTURE_DURATION,
                store=True
            )
        except Exception as e:
            print(f"    ❌ Capture error: {e}")
            print(f"       Make sure you're running with sudo and interface '{self.interface}' exists")
            return None

        # Wait for playback thread to complete
        playback_thread.join(timeout=2)

        # Check for playback errors
        if playback_error['error']:
            print(f"    ⚠️  Playback error: {playback_error['error']}")
            # Continue anyway - we still captured packets

        print(f"    📦 Total packets captured: {len(packets)}")

        if packets and len(packets) > 0:
            # Save captured packets
            try:
                wrpcap(pcap_filename, packets)
                file_size = os.path.getsize(pcap_filename)
                print(f"    ✅ Saved: {pcap_filename} ({file_size / 1024:.1f} KB)")
                return pcap_filename
            except Exception as e:
                print(f"    ❌ Error saving PCAP: {e}")
                return None
        else:
            print(f"    ⚠️  No packets captured!")
            return None

    def generate_dataset(self):
        try:
            self.setup_spotify_client()

            if MUSIC_PLAYLISTS:
                self.fetch_playlist_tracks()

            if PODCAST_PLAYLISTS:
                self.fetch_podcast_episodes()

            total_items = len(self.music_tracks) + len(self.podcast_episodes)

            if total_items == 0:
                print("\n❌ No content found! Add playlist IDs to MUSIC_PLAYLISTS or PODCAST_PLAYLISTS")
                return

            print("\n" + "=" * 70)
            print("COLLECTION SUMMARY")
            print("=" * 70)
            print(f"Music tracks:       {len(self.music_tracks)}")
            print(f"Podcast episodes:   {len(self.podcast_episodes)}")
            print(f"Total items:        {total_items}")
            print(f"Capture duration:   {CAPTURE_DURATION} seconds per item")
            print(f"Estimated time:     ~{total_items * (CAPTURE_DURATION + 7) // 60} minutes")
            print(f"\n📝 Note: Capturing ALL traffic (no IP filtering)")
            print(f"    Feature extraction will handle Spotify IP filtering later")
            print("=" * 70)

            input("\n👉 Press Enter to start capturing...")

            print("\n" + "=" * 70)
            print("STARTING CAPTURE")
            print("=" * 70)

            captured_count = 0
            failed_count = 0
            current_item = 0

            # Capture music tracks
            for track in self.music_tracks:
                result = self.capture_content_traffic("music", track, current_item, total_items)
                if result:
                    captured_count += 1
                else:
                    failed_count += 1

                current_item += 1
                if current_item < total_items:
                    print("    ⏳ Waiting 5 seconds before next capture...")
                    time.sleep(5)

            # Capture podcast episodes
            for episode in self.podcast_episodes:
                result = self.capture_content_traffic("podcast", episode, current_item, total_items)
                if result:
                    captured_count += 1
                else:
                    failed_count += 1

                current_item += 1
                if current_item < total_items:
                    print("    ⏳ Waiting 5 seconds before next capture...")
                    time.sleep(5)

            # Summary
            print("\n" + "=" * 70)
            print("🎉 CAPTURE COMPLETE")
            print("=" * 70)
            print(f"✅ Successfully captured: {captured_count}/{total_items} items")
            if failed_count > 0:
                print(f"❌ Failed captures:       {failed_count}/{total_items} items")
            print(f"📁 PCAPs saved to:        {self.pcap_dir}/")
            print("\n📌 Next steps:")
            print("   1. Run the feature extractor to process PCAPs")
            print("   2. The extractor will filter by Spotify IPs from spotify_ips.json")
            print("=" * 70)

        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys

    try:
        print("=" * 70)
        print("🎵 SPOTIFY PCAP CAPTURE - SCAPY MODE")
        print("=" * 70)

        # Check authentication BEFORE requiring root
        if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI]):
            print("\n❌ ERROR: Spotify credentials not found in .env file!")
            print("Please create a .env file with:")
            print("  SPOTIFY_CLIENT_ID=your_client_id")
            print("  SPOTIFY_CLIENT_SECRET=your_client_secret")
            print("  SPOTIFY_REDIRECT_URI=your_redirect_uri")
            sys.exit(1)

        # Check if we need to authenticate first
        sudo_user = os.environ.get('SUDO_USER')
        if sudo_user:
            import pwd

            user_home = pwd.getpwnam(sudo_user).pw_dir
            cache_path = os.path.join(user_home, '.cache', 'spotify-pcap-token')
        else:
            cache_path = '.cache-spotify-pcap-token'


        # Handle --auth flag for initial authentication
        if len(sys.argv) > 1 and sys.argv[1] == '--auth':
            if os.geteuid() == 0:
                print("\n❌ ERROR: Don't run --auth with sudo!")
                print("Run as normal user: python3 capture.py --auth")
                sys.exit(1)

            print("\n📋 AUTHENTICATING WITH SPOTIFY")
            print("=" * 70)
            print("This will open a browser window for Spotify login.")
            print("After logging in, the token will be cached for sudo use.")
            print("=" * 70)
            input("\nPress Enter to continue...")

            scope = "user-modify-playback-state user-read-playback-state"
            sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope=scope,
                cache_path=cache_path
            ))

            # Test the connection
            user = sp.current_user()
            print(f"\n✅ Successfully authenticated as: {user['display_name']}")
            print(f"📁 Token cached at: {cache_path}")
            print("\nYou can now run the capture with sudo:")
            print("  sudo python3 capture.py")
            sys.exit(0)

        print("\n📋 PREREQUISITES:")
        print("  ✓ Running as root (for packet capture)")
        print("  ✓ Spotify credentials configured in .env file")
        print("  ✓ Authenticated with Spotify (token cached)")
        print("\n📝 WORKFLOW:")
        print("  1. This script connects to your running Spotify via API")
        print("  2. Uses Scapy to capture ALL traffic during playback")
        print("  3. Organizes captures by genre in pcap/ directory")
        print("  4. Feature extractor later filters by Spotify IPs")
        print("\n⚠️  REQUIREMENTS:")
        print("  • Spotify must be OPEN and RUNNING")
        print("  • Device must be active (play something first)")
        print("=" * 70)

        print("\n⚙️  Configure capture:")
        while True:
            try:
                tracks = int(input("  Tracks per playlist (default 10): ").strip() or "10")
                episodes = int(input("  Episodes per playlist (default 10): ").strip() or "10")
                if tracks > 0 and episodes > 0:
                    break
                print("  ❌ Enter positive numbers")
            except ValueError:
                print("  ❌ Enter valid numbers")

        interface = input("  Network interface (default: enp0s3): ").strip() or "enp0s3"

        print("\n" + "=" * 70)
        print("⚠️  FINAL CHECK:")
        print("  1. Is Spotify OPEN and RUNNING?")
        print("  2. Are you logged in?")
        print("  3. Have you played something to activate the device?")
        print("=" * 70)
        input("Press Enter when ready to start capturing...")

        generator = SpotifyPcapCapture(
            interface=interface,
            tracks_per_playlist=tracks,
            episodes_per_playlist=episodes
        )
        generator.generate_dataset()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)