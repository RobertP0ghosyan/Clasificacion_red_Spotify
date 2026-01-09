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
CAPTURE_DURATION = 15  # seconds

# List of song URIs to capture (replace with your test songs)
SONG_URIS = [
    "spotify:track:5SudOD9R1Of6CsJVWZy6CQ",
    "spotify:track:4oVO4fGNRRvEn0CRuFO4qv",
    "spotify:track:6hpuesKPNa3WhV48O7Fa47"
]

STREAMING_QUALITY = [
    'low',
    'normal',
    'high',
    'very-high'
]


class SpotifyDatasetGenerator:
    def __init__(self, audio_quality, interface="ens33"):
        self.driver = None
        self.spotify_client = None
        self.captured_data = []
        self.interface = interface
        self.dataset_dir = 'dataset'
        self.pcap_captures_dir = 'pcap'
        self.audio_quality = audio_quality

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.pcap_captures_dir, exist_ok=True)

        self.dataset_file=f"{self.dataset_dir}/spotify_traffic_dataset_{audio_quality}.csv"
                
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
    
    def packet_callback(self, packet):
        """Callback function for packet capture"""
        arrival_time = packet.time
        payload_size = len(packet)
        self.current_capture.append((arrival_time, payload_size))
    
    def capture_song_traffic(self, song_uri, song_index):
        """Capture network traffic for a specific song"""
        print(f"Capturing traffic for song {song_index + 1}/{len(SONG_URIS)}...")
        
        # Start playback using Spotipy
        try:
            self.spotify_client.start_playback(uris=[song_uri])
            print(f"Started playback: {song_uri}")
        except Exception as e:
            print(f"Error starting playback: {e}")
            print("Attempting to continue with current playback...")
        
        # Initialize capture list for this song
        self.current_capture = []
        
        # Start packet sniffing
        print(f"   Sniffing packets for {CAPTURE_DURATION} seconds...")
        try:
            captured_packets = sniff(
                iface=self.interface,
                filter="tcp port 443",
                prn=self.packet_callback,
                timeout=CAPTURE_DURATION,
                store=True
            )

            wrpcap(f"{self.pcap_captures_dir}/{time.strftime("%d-%m-%Y-%H%M%S")}_{song_uri}_{self.audio_quality}.pcap", captured_packets)
        except PermissionError:
            print("ERROR: Permission denied. Please run script with sudo/admin privileges")
            raise
        
        print(f"    Captured {len(self.current_capture)} packets")
        
        # Store the capture
        return self.current_capture.copy()
    
    def save_dataset(self, new_data):
        """Save or append data to CSV file"""
        print("Saving dataset...")
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(self.dataset_file)
        
        # Open CSV file in append mode
        with open(self.dataset_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write headers if new file
            if not file_exists:
                writer.writerow(['song_id', 'packet_arrival_time', 'payload_size'])
                print("   Creating new dataset file with headers")
            else:
                print(f"   Appending to existing dataset")
            
            # Write data for each song
            total_packets = 0
            
            for song_idx, song_capture in enumerate(new_data):
                song_id = SONG_URIS[song_idx]
                
                for arrival_time, payload_size in song_capture:
                    writer.writerow([
                        song_id,
                        arrival_time,
                        payload_size
                    ])
                    total_packets += 1
        
        print(f"    Saved {total_packets} packets from {len(new_data)} songs")
        print(f"    File: {self.dataset_file}")
    
    def generate_dataset(self):
        """Main method to generate the dataset"""
        try:
            # Setup
            self.setup_spotify_client()
            
            # Capture data for each song
            captured_songs = []
            for i, song_uri in enumerate(SONG_URIS):
                song_data = self.capture_song_traffic(song_uri, i)
                captured_songs.append(song_data)
            
            # Save the dataset
            self.save_dataset(captured_songs)
            
            print("\n" + "="*50)
            print("Dataset generation complete!")
            print(f"Total songs captured: {len(captured_songs)}")
            for i, song_data in enumerate(captured_songs):
                print(f"  Song {i+1}: {len(song_data)} packets")
            print("="*50)
            
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
    print("3. Update SONG_URIS with your test songs")
    print("4. Install required packages:")
    print("   pip install -r requirements.txt")
    print("=" * 50 + "\n")
    
    print("\nAvailable qualities:")
    print("1. Low")
    print("2. Normal")
    print("3. High")
    print("4. Very high")

    quality_idx = int(input("\nSelect the quality configured in Spotify (1-4):").strip())

    print(f"Selected quality: {STREAMING_QUALITY[quality_idx - 1]}")

    input("\nStart the Spotify player in a device and press Enter to start data collection...")

    
    generator = SpotifyDatasetGenerator(audio_quality=STREAMING_QUALITY[quality_idx - 1])
    generator.generate_dataset()