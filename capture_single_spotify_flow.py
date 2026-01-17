#!/usr/bin/env python3
import time
import os
from scapy.all import sniff, wrpcap
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import json

load_dotenv()

# =========================
# Spotify credentials
# =========================

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

CONFIG_FILE = "capture_config.json"


class SpotifySingleCapture:
    def __init__(self, interface, pcap_save_dir, audio_quality, capture_duration):
        self.interface = interface
        self.pcap_save_dir = pcap_save_dir
        self.audio_quality = audio_quality
        self.capture_duration = capture_duration
        self.spotify_client = None

        os.makedirs(self.pcap_save_dir, exist_ok=True)

    def setup_spotify_client(self):
        scope = "user-modify-playback-state user-read-playback-state"
        self.spotify_client = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope=scope
            )
        )

    def capture(self, song_uri):
        # Start playback
        devices = self.spotify_client.devices()["devices"]
        if not devices:
            raise RuntimeError("No active Spotify device found")

        device_id = devices[0]["id"]
        self.spotify_client.start_playback(uris=[song_uri], device_id=device_id)

        print(f"[+] Playing {song_uri}")
        print(f"[+] Capturing traffic for {self.capture_duration} seconds")

        packets = sniff(
            iface=self.interface,
            timeout=self.capture_duration,
            store=True
        )

        ts = time.strftime("%d-%m-%Y-%H%M%S")
        out_file = f"{ts}_spotify_{song_uri}_{self.audio_quality}.pcap"
        out_path = os.path.join(self.pcap_save_dir, out_file)

        wrpcap(out_path, packets)

        print(f"[✓] Capture finished → {out_path}")
        print(f"[i] Packets captured: {len(packets)}")


# =========================
# Main
# =========================

if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE):
        raise Exception(f"Config file '{CONFIG_FILE}' not found")

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    print("\nAvailable Spotify URIs:")
    for i, uri in enumerate(config["song_uris"]):
        print(f"  {i+1}. {uri}")

    idx = int(input("\nSelect ONE song URI: ").strip()) - 1
    song_uri = config["song_uris"][idx]

    print("\nAvailable qualities:")
    for i, q in enumerate(config["streaming_qualities"]):
        print(f"  {i+1}. {q}")

    qidx = int(input("\nSelect configured quality: ").strip()) - 1
    quality = config["streaming_qualities"][qidx]

    input("\nMake sure Spotify is open on ONE device. Press Enter to start...")

    capturer = SpotifySingleCapture(
        interface=config["interface"],
        pcap_save_dir=config["pcap_save_dir"],
        audio_quality=quality,
        capture_duration=config["capture_duration"]
    )

    capturer.setup_spotify_client()
    capturer.capture(song_uri)
