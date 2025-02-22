import os
import sys
import logging
from mutagen.mp3 import MP3
from mutagen.id3 import ID3NoHeaderError
import musicbrainzngs

# Configure logging
logging.basicConfig(
    filename='metadata.log',
    level=logging.DEBUG,  # Set to DEBUG to capture all log levels
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_metadata_from_musicbrainz(title, artist):
    # Attempt to fetch metadata from MusicBrainz using title and artist
    try:
        logging.info(f"Searching for '{title}' by '{artist}' in MusicBrainz.")
        results = musicbrainzngs.search_recordings(query=title, artist=artist)

        if results['recording-list']:
            recording = results['recording-list'][0]
            title = recording.get('title', 'Unknown')
            artist = ', '.join(artist['name'] for artist in recording.get('artist-credit', []))
            album = recording.get('release-list', [{}])[0].get('title', 'Unknown')
            genre = 'Unknown'  # Genre may not be available in the recording data
            year = recording.get('first-release-date', 'Unknown').split('-')[0]  # Extract year

            logging.info(f"Fetched metadata for '{title}' by '{artist}' from MusicBrainz.")
            return title, artist, album, genre, year
        else:
            logging.warning(f"No metadata found for '{title}' by '{artist}' in MusicBrainz.")
            return None, None, None, None, None

    except Exception as e:
        logging.error(f"Error fetching metadata from MusicBrainz for '{title}' by '{artist}': {e}")
        return None, None, None, None, None

def print_missing_metadata(file_path):
    try:
        audio = MP3(file_path)
        logging.info(f"Processing file: {file_path}")

        # Extract metadata
        title = audio.get('TIT2', None)
        artist = audio.get('TPE1', None)
        album = audio.get('TALB', None)
        genre = audio.get('TCON', None)
        year = audio.get('TDRC', None)

        missing_metadata = {}

        # Check for missing metadata
        if not title:
            missing_metadata['Title'] = 'Unknown'
        if not artist:
            missing_metadata['Artist'] = 'Unknown'
        if not album:
            missing_metadata['Album'] = 'Unknown'
        if not genre:
            missing_metadata['Genre'] = 'Unknown'
        if not year:
            missing_metadata['Year'] = 'Unknown'

        # Print missing metadata if any
        if missing_metadata:
            logging.info(f"Missing metadata found in {file_path}: {missing_metadata}")
            print("Missing Metadata:")
            for key, value in missing_metadata.items():
                print(f"  {key}: {value}")

            # Ask if the user wants to search for missing metadata
            if not title and not artist:
                search_option = input("Do you want to search for this metadata on MusicBrainz? (yes/no): ").strip().lower()
                if search_option == 'yes':
                    fetched_title, fetched_artist, fetched_album, fetched_genre, fetched_year = fetch_metadata_from_musicbrainz(title, artist)
                    if fetched_title:
                        print(f"Fetched Metadata from MusicBrainz:")
                        print(f"  Title: {fetched_title}")
                        print(f"  Artist: {fetched_artist}")
                        print(f"  Album: {fetched_album}")
                        print(f"  Genre: {fetched_genre}")
                        print(f"  Year: {fetched_year}")
            else:
                # If either title or artist is present, attempt to fetch metadata
                if title and artist:
                    fetched_title, fetched_artist, fetched_album, fetched_genre, fetched_year = fetch_metadata_from_musicbrainz(title, artist)
                    if fetched_title:
                        print(f"Fetched Metadata from MusicBrainz:")
                        print(f"  Title: {fetched_title}")
                        print(f"  Artist: {fetched_artist}")
                        print(f"  Album: {fetched_album}")
                        print(f"  Genre: {fetched_genre}")
                        print(f"  Year: {fetched_year}")

        print("\n")
        
    except ID3NoHeaderError:
        logging.warning(f"No ID3 header found in {file_path}.")
        print(f"No ID3 header found in {file_path}.")
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")

