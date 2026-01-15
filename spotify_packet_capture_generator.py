import time
import os
from scapy.all import sniff, wrpcap
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import json

load_dotenv()

# Configuration
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

CONFIG_FILE = "capture_config.json"


class SpotifyNetworkCaptureGenerator:
    def __init__(self,
                 interface,
                 pcap_save_dir,
                 audio_quality,
                 song_uris,
                 capture_duration,
                 num_sessions):

        self.spotify_client = None
        self.interface = interface
        self.pcap_save_dir = pcap_save_dir
        self.audio_quality = audio_quality
        self.song_uris = song_uris
        self.capture_duration = capture_duration
        self.num_sessions = num_sessions

        os.makedirs(self.pcap_save_dir, exist_ok=True)
                
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
        print("    Spotify client authenticated")
    
    def capture_song_traffic(self, uri):
        """Capture network traffic for a specific song/podcast"""
        
        # Start playback using Spotipy
        try:
            # check for devices running spotify
            devices = self.spotify_client.devices()["devices"]
            if devices:
                self.spotify_client.start_playback(uris=[uri], device_id=devices[0]["id"])
                print(f"Started playback: {uri}")
            else:
                print("No device detected")
                exit(1)
                
        except Exception as e:
            print(f"Error starting playback: {e}")
            print("Attempting to continue with current playback...")
        
        
        # Start packet sniffing
        print(f"   Sniffing packets for {self.capture_duration} seconds...")
        try:
            captured_packets = sniff(
                iface=self.interface,
                timeout=self.capture_duration,
                store=True
            )

            wrpcap(f"{self.pcap_save_dir}/{time.strftime("%d-%m-%Y-%H%M%S")}_{uri}_{self.audio_quality}.pcap", captured_packets)
        except PermissionError:
            print("ERROR: Permission denied. Please run script with sudo/admin privileges")
            raise
        
        print(f"    Captured {len(captured_packets)} packets")
    
    def generate_capture(self):
        """Main method to generate the dataset"""
        try:
            # Setup
            self.setup_spotify_client()
            for _ in range(self.num_sessions):
                # Capture data for each song
                for song_uri in self.song_uris:
                    self.capture_song_traffic(song_uri)
                
        
        except KeyboardInterrupt:
            print("\n\nCapture interrupted by user")
        except Exception as e:
            print(f"\n\nError during capture: {e}")
            raise


if __name__ == "__main__":
    print("=" * 50)
    print("Spotify Traffic Dataset Generator")
    print("=" * 50)
    print("\nIMPORTANT NOTES:")
    print("1. Run this script with sudo/administrator privileges")
    print("2. Update SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in the env file")
    print(f"3. Configure everything in {CONFIG_FILE}")
    print("4. Install required packages:")
    print("   pip install -r requirements.txt")
    print("=" * 50 + "\n")

    # Load config
    if not os.path.exists(CONFIG_FILE):
        raise Exception(f"Config file '{CONFIG_FILE}' not found!")
    
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    streaming_qualities = config["streaming_qualities"]

    print("\nAvailable qualities:")
    for i, quality in enumerate(streaming_qualities):
        print(f"    {i+1}. {quality}")

    quality_idx = int(input(f"\nSelect the quality configured in Spotify (1-{len(streaming_qualities)}):").strip())

    print(f"    Selected quality: {streaming_qualities[quality_idx - 1]}")

    input("\nStart Spotify on ONE device and press Enter to start data collection...")

    
    generator = SpotifyNetworkCaptureGenerator(interface=config["interface"],
                                               pcap_save_dir=config["pcap_save_dir"],
                                               audio_quality=streaming_qualities[quality_idx - 1],
                                               song_uris=config["song_uris"],
                                               capture_duration=config["capture_duration"],
                                               num_sessions=config["num_sessions"])
    generator.generate_capture()