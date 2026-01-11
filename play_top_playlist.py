import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import time

# Load .env file
load_dotenv()

# Configuration
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

# ADD YOUR PLAYLISTS HERE (just the ID from the URL)
MUSIC_PLAYLISTS = [
    # "7IddiFVjAJbTLniq82Vusj",  # Pink Floyd Best Of
    # "5HpkkM0bOPDUgLcho7nCoZ", #Tame Impala best songs
    # "28nxGp2hLho3BA0dX3cb5P", #Best of Radiohead
    # "6Fs9lBMpHdqjvQ6wCPDnKc", #Peak Kannye
    # "5lBSVt2KMhM5EdSm6WSxpg", #thss iis kendrick lamar  (hip hop)
    # "3IffYurXS0a9WC3SikI4TV", #travis(rap) chill vibes
    # "35kZMub9UFGSheeghSXBfw", #(neo-psychedelic) Best of Tame Impala
    # -------
    # "0U3ACsVhROtNwwacDmhcuR", #(Progressive Rock) 25 King Crimson
    # "4yebu47SKvUq8aWmTu1cRc",# david bowie Art Rock
    #
    # "4gHuAdOjAZHMb6WYKQhbLD",  # (neo-psychedelic) mgmt
    "1mkinKlTq2OV9MCE5Nkpp9",
    "10PXjjuLhwtYRZtJkgixLO",
    "6Sv7aZ1fHZVEWfGdhqWn87",
    "0yskWBwX31blZR9bVCBZTL",

]

# ADD YOUR PODCAST SHOWS HERE (just the ID from the URL)
PODCAST_SHOWS = [
    # "5CnDmMUG0S5bSSw612fs8C",  # Bad Friends
]

# How long to play each track/episode (in seconds)
PLAYBACK_DURATION = 15


def get_artist_genre(sp, artist_id):
    """Fetch primary genre from artist"""
    try:
        artist = sp.artist(artist_id)
        genres = artist.get("genres", [])
        print(genres)
        return genres[0] if genres else "unknown"
    except Exception as e:
        print(f" Genre fetch failed: {e}")
        return "unknown"

def get_playlist_tracks(sp, playlist_id, limit=50):
    """Get top tracks from a playlist"""
    print(f"\nFetching playlist: {playlist_id}")
    playlist = sp.playlist(playlist_id)
    print(f"  ✓ {playlist['name']} ({playlist['tracks']['total']} tracks) ")

    results = sp.playlist_tracks(playlist_id, limit=limit)
    tracks = []

    for item in results['items']:
        track = item.get('track')
        if not track:
            continue  # skip removed/local tracks

        artist = track['artists'][0]
        artist_id = artist['id']
        genre = get_artist_genre(sp, artist_id)

        tracks.append({
            'uri': track['uri'],
            'name': track['name'],
            'artist': artist['name'],          # metadata only
            'type': 'music',
            'primary_genre': genre,
            'source': playlist['name']
        })

    print(f"    Got {len(tracks)} tracks")
    print(f"    Genre: {genre}")
    return tracks



def get_show_episodes(sp, show_id, limit=50):
    """Get top episodes from a podcast show"""
    print(f"\nFetching show: {show_id}")
    show = sp.show(show_id)
    print(f"  ✓ {show['name']} ({show['total_episodes']} episodes)")

    results = sp.show_episodes(show_id, limit=limit)
    episodes = []

    for item in results['items']:
        episodes.append({
            'uri': item['uri'],
            'name': item['name'],
            'artist': show['publisher'],
            'type': 'podcast',
            'source': show['name']
        })

    print(f"    Got {len(episodes)} episodes")
    return episodes


def play_content(sp, content, index, total):
    """Play a single track or episode"""
    content_type = "🎵" if content['type'] == 'music' else "🎙️"
    print(f"\n{content_type} [{index}/{total}] {content['artist']} - {content['name']}")
    print(f"   From: {content['source']}")

    try:
        sp.start_playback(uris=[content['uri']], position_ms=0)

        # Show progress bar
        for i in range(PLAYBACK_DURATION):
            time.sleep(1)
            progress = "█" * (i + 1) + "░" * (PLAYBACK_DURATION - i - 1)
            print(f"\r  [{progress}] {i + 1}/{PLAYBACK_DURATION}s", end="", flush=True)

        print("\n  ✓ Done")
        sp.pause_playback()
        return True

    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False


def main():
    print("=" * 70)
    print("SPOTIFY PLAYLIST & PODCAST PLAYER")
    print("=" * 70)
    print()

    # Setup Spotify client
    print("Connecting to Spotify...")
    scope = "user-modify-playback-state user-read-playback-state"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=scope
    ))
    print("✓ Connected!\n")

    # Check for active device
    print("Checking for active Spotify device...")
    devices = sp.devices()
    if not devices['devices']:
        print("✗ No active Spotify device found!")
        print("  Please open Spotify on a device and try again.")
        return

    # Collect all content
    print("=" * 70)
    print("COLLECTING CONTENT")
    print("=" * 70)

    all_content = []

    # Get music from playlists
    if MUSIC_PLAYLISTS:
        print(f"\n📀 Fetching {len(MUSIC_PLAYLISTS)} music playlist(s)...")
        for playlist_id in MUSIC_PLAYLISTS:
            tracks = get_playlist_tracks(sp, playlist_id, limit=50)
            all_content.extend(tracks)

    # Get podcast episodes
    if PODCAST_SHOWS:
        print(f"\n🎙️  Fetching {len(PODCAST_SHOWS)} podcast show(s)...")
        for show_id in PODCAST_SHOWS:
            episodes = get_show_episodes(sp, show_id, limit=50)
            all_content.extend(episodes)

    # Summary
    print("\n" + "=" * 70)
    print("CONTENT SUMMARY")
    print("=" * 70)

    music_count = sum(1 for c in all_content if c['type'] == 'music')
    podcast_count = sum(1 for c in all_content if c['type'] == 'podcast')

    print(f"🎵 Music tracks: {music_count}")
    print(f"🎙️  Podcast episodes: {podcast_count}")
    print(f"📊 Total items: {len(all_content)}")
    print(f"⏱️  Total time: ~{len(all_content) * PLAYBACK_DURATION // 60} minutes")

    if not all_content:
        print("\n✗ No content found! Add playlists/shows to the lists at the top.")
        return

    # Ask to start
    print("\n" + "=" * 70)
    response = input(f"Play all {len(all_content)} items ({PLAYBACK_DURATION}s each)? (y/n): ")

    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Play everything
    print("\n" + "=" * 70)
    print("STARTING PLAYBACK")
    print("=" * 70)

    successful = 0
    for idx, content in enumerate(all_content, 1):
        if play_content(sp, content, idx, len(all_content)):
            successful += 1

        # Pause between items
        if idx < len(all_content):
            time.sleep(2)

    # Final summary
    print("\n" + "=" * 70)
    print("🎉 PLAYBACK COMPLETE!")
    print("=" * 70)
    print(f"✓ Successfully played: {successful}/{len(all_content)}")
    print(f"✗ Failed: {len(all_content) - successful}")
    print("=" * 70)


if __name__ == "__main__":
    main()