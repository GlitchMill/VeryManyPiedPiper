import os
import re
import time
import threading
import logging
import numpy as np
import librosa
import soundfile as sf
import musicbrainzngs
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from flask_socketio import SocketIO, emit
from io import BytesIO
from PIL import Image
# For 3D plotting
import pandas as pd
import plotly.express as px

# Configure logging to write to a file (no console output)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log", mode="a", encoding="utf-8")]
)

app = Flask(__name__)
socketio = SocketIO(app)
musicbrainz_cache = {}

SONGS_FOLDER = 'songs'
ALBUM_ART_FOLDER = os.path.join('static', 'album_art')
DEFAULT_ALBUM_ART = '/static/album_art/default.jpg'

# Ensure album art folder exists
if not os.path.exists(ALBUM_ART_FOLDER):
    os.makedirs(ALBUM_ART_FOLDER)
    logging.info(f"Created album art folder: {ALBUM_ART_FOLDER}")

# Initialize MusicBrainz user agent
musicbrainzngs.set_useragent("MyMusicApp", "1.0", "you@example.com")
logging.info("MusicBrainz user agent set.")

def clean_filename(filename):
    """Clean the filename to create a query string for MusicBrainz."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\[.*?\]', '', name)
    name = name.replace('_', ' ').replace('-', ' ')
    cleaned = name.strip()
    logging.debug(f"Cleaned filename '{filename}' to '{cleaned}'")
    return cleaned

def get_metadata_from_musicbrainz(filename):
    """
    Given a filename, query MusicBrainz for the best matching recording.
    Uses a cache to avoid repeated API calls.
    Returns (title, artist) if found; otherwise (None, None).
    """
    query = clean_filename(filename)
    if query in musicbrainz_cache:
        logging.info(f"Using cached MusicBrainz result for: {query}")
        return musicbrainz_cache[query]
    logging.info(f"Querying MusicBrainz for: {query}")
    try:
        result = musicbrainzngs.search_recordings(query=query, limit=1)
        recordings = result.get('recording-list', [])
        if recordings:
            rec = recordings[0]
            title = rec.get('title', None)
            artist = 'Unknown'
            if 'artist-credit' in rec and rec['artist-credit']:
                artist = rec['artist-credit'][0].get('name', 'Unknown')
            musicbrainz_cache[query] = (title, artist)
            logging.info(f"MusicBrainz found: Title: {title}, Artist: {artist}")
            return title, artist
    except Exception as e:
        logging.error(f"Error fetching metadata from MusicBrainz for {filename}: {e}")
    musicbrainz_cache[query] = (None, None)
    return None, None

def compute_embedding(file_path):
    """
    Compute a simple embedding for the audio file using the mean of the first 3 MFCC coefficients.
    """
    try:
        y, sr = sf.read(file_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        embedding = np.mean(mfcc[:3, :], axis=1)
        logging.debug(f"Computed embedding for {file_path}: {embedding}")
        return embedding
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return np.zeros(3)

def extract_and_save_album_art(file_path, song_id):
    """Extract album art from an audio file and save thumbnail."""
    album_art_data = None
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(file_path)
        if audio is not None and audio.tags is not None:
            # For MP3 files: check for APIC frames
            apic_keys = [k for k in audio.tags.keys() if k.startswith('APIC')]
            if apic_keys:
                album_art_data = audio.tags[apic_keys[0]].data
            if not album_art_data and hasattr(audio, 'pictures') and audio.pictures:
                # For FLAC files: check for pictures
                album_art_data = audio.pictures[0].data
    except Exception as e:
        print(f"Error extracting album art: {e}")

    if album_art_data:
        album_art_path = os.path.join(ALBUM_ART_FOLDER, f"album_{song_id}.jpg")
        try:
            img = Image.open(BytesIO(album_art_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((250, 250), Image.LANCZOS)
            img.save(album_art_path, "JPEG")
            print(f"Album art saved to {album_art_path}")
            return f"/static/album_art/album_{song_id}.jpg"  # Ensure this returns the correct URL
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        print("No album art found.")
        return DEFAULT_ALBUM_ART  # If no album art, return the default image URL

def load_songs_from_folder(directory):
    """
    Scan the given folder (non-recursively) for .mp3 and .flac files,
    extract metadata (or fetch via fuzzy search), compute embeddings, and extract album art.
    """
    logging.info(f"Scanning directory for songs: {directory}")
    songs_list = []
    id_counter = 1
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.mp3', '.flac')):
            logging.info(f"Processing file: {file_path}")
            embedding = compute_embedding(file_path)
            base_title = os.path.splitext(file)[0]
            title = base_title
            artist = "Unknown"
            try:
                from mutagen import File as MutagenFile
                audio = MutagenFile(file_path)
                if audio is not None and audio.tags is not None:
                    if 'TIT2' in audio:
                        title = audio['TIT2'].text[0]
                    if 'TPE1' in audio:
                        artist = audio['TPE1'].text[0]
            except Exception as e:
                logging.error(f"Metadata extraction error for {file}: {e}")
            if title.lower() == base_title.lower() or artist == "Unknown":
                mb_title, mb_artist = get_metadata_from_musicbrainz(file)
                if mb_title:
                    title = mb_title
                if mb_artist:
                    artist = mb_artist
            album_art_url = extract_and_save_album_art(file_path, id_counter)
            songs_list.append({
                'id': id_counter,
                'title': title,
                'artist': artist,
                'file': os.path.relpath(file_path, SONGS_FOLDER),
                'embedding': embedding,
                'album_art': album_art_url,
                'album': None  # Will be set later if file is inside an album folder
            })
            id_counter += 1
    logging.info(f"Found {len(songs_list)} songs in {directory}")
    return songs_list

def load_all_songs_from_folder(root):
    """
    Recursively scan the root folder (including subdirectories) for audio files.
    Returns a list of song dictionaries.
    """
    logging.info(f"Recursively scanning for songs in: {root}")
    songs_list = []
    id_counter = 1
    for dirpath, dirnames, filenames in os.walk(root):
        for file in filenames:
            if file.lower().endswith(('.mp3', '.flac')):
                file_path = os.path.join(dirpath, file)
                logging.debug(f"Processing file: {file_path}")
                embedding = compute_embedding(file_path)
                base_title = os.path.splitext(file)[0]
                title = base_title
                artist = "Unknown"
                try:
                    from mutagen import File as MutagenFile
                    audio = MutagenFile(file_path)
                    if audio is not None and audio.tags is not None:
                        if 'TIT2' in audio:
                            title = audio['TIT2'].text[0]
                        if 'TPE1' in audio:
                            artist = audio['TPE1'].text[0]
                except Exception as e:
                    logging.error(f"Metadata extraction error for {file}: {e}")
                if title.lower() == base_title.lower() or artist == "Unknown":
                    mb_title, mb_artist = get_metadata_from_musicbrainz(file)
                    if mb_title:
                        title = mb_title
                    if mb_artist:
                        artist = mb_artist
                album_art_url = extract_and_save_album_art(file_path, id_counter)
                # Set album name if file is in a subdirectory (other than root)
                rel_dir = os.path.relpath(dirpath, root)
                album = rel_dir if rel_dir != '.' else None
                songs_list.append({
                    'id': id_counter,
                    'title': title,
                    'artist': artist,
                    'file': os.path.relpath(file_path, SONGS_FOLDER),
                    'embedding': embedding,
                    'album_art': album_art_url,
                    'album': album
                })
                id_counter += 1
    logging.info(f"Recursively found {len(songs_list)} songs in {root}")
    return songs_list

def load_albums_from_folder(root):
    """
    Look for immediate subdirectories in the root and treat each as an album.
    Loads songs (non-recursively) from each album folder.
    """
    logging.info(f"Scanning for album folders in: {root}")
    albums = []
    for item in os.listdir(root):
        album_path = os.path.join(root, item)
        if os.path.isdir(album_path):
            album_songs = load_songs_from_folder(album_path)
            if album_songs:
                album = {
                    'name': item,
                    'songs': album_songs,
                    'album_art': album_songs[0]['album_art'] if album_songs else DEFAULT_ALBUM_ART
                }
                albums.append(album)
                logging.info(f"Found album: {item} with {len(album_songs)} songs")
    return albums

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def get_recommendations(song_id, top_n=2):
    """
    Given a song id, compute cosine similarity between its embedding and all other songs,
    then return the top_n similar songs from the global list.
    """
    selected_song = next((song for song in songs_all if song['id'] == song_id), None)
    if not selected_song:
        logging.warning(f"Song ID {song_id} not found for recommendations.")
        return []
    similarities = []
    for song in songs_all:
        if song['id'] != song_id:
            sim = cosine_similarity(selected_song['embedding'], song['embedding'])
            similarities.append((sim, song))
    similarities.sort(key=lambda x: x[0], reverse=True)
    logging.info(f"Found {len(similarities)} similar songs for song ID {song_id}")
    return [song for sim, song in similarities[:top_n]]

# Initial scan on startup:
logging.info("Performing initial scan of songs folder.")
songs_all = load_all_songs_from_folder(SONGS_FOLDER)
individual_songs = load_songs_from_folder(SONGS_FOLDER)
albums = load_albums_from_folder(SONGS_FOLDER)
logging.info(f"Initial scan complete. Total songs: {len(songs_all)}. Albums found: {len(albums)}.")

def update_songs_periodically():
    """
    Periodically rescan the SONGS_FOLDER and update the global lists.
    Emit a websocket event to update connected clients if data changes.
    """
    global songs_all, individual_songs, albums
    while True:
        logging.info("Rescanning songs folder for updates...")
        new_songs_all = load_all_songs_from_folder(SONGS_FOLDER)
        new_individual_songs = load_songs_from_folder(SONGS_FOLDER)
        new_albums = load_albums_from_folder(SONGS_FOLDER)
        songs_all = new_songs_all
        individual_songs = new_individual_songs
        albums = new_albums
        logging.info(f"Updated song list. Total songs: {len(songs_all)}. Albums found: {len(albums)}.")
        # Emit websocket event to notify clients
        socketio.emit('update_data', {'msg': 'Data updated'}, broadcast=True)
        time.sleep(60)  # Rescan every 60 seconds

# Start the background update thread
update_thread = threading.Thread(target=update_songs_periodically, daemon=True)
update_thread.start()
logging.info("Background update thread started.")

@app.route('/')
def index():
    """Render the main page showing Albums and individual Songs."""
    return render_template('index.html', albums=albums, songs=individual_songs)

@app.route('/album/<album_name>')
def album_view(album_name):
    """
    Render a page for a specific album (i.e. a subdirectory in SONGS_FOLDER).
    """
    album_path = os.path.join(SONGS_FOLDER, album_name)
    if not os.path.isdir(album_path):
        logging.error(f"Album not found: {album_name}")
        abort(404)
    album_songs = load_songs_from_folder(album_path)
    logging.info(f"Rendering album view for {album_name} with {len(album_songs)} songs.")
    return render_template('album.html', album_name=album_name, songs=album_songs)

@app.route('/songs/<path:filename>')
def serve_song(filename):
    """Serve the audio files from the SONGS_FOLDER."""
    logging.debug(f"Serving song file: {filename}")
    return send_from_directory(SONGS_FOLDER, filename)

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    API endpoint to get recommendations based on a song id.
    Example: /recommend?song_id=1
    """
    song_id = request.args.get('song_id', type=int)
    recommendations = get_recommendations(song_id)
    rec_list = [{
        'id': s['id'],
        'title': s['title'],
        'artist': s['artist'],
        'album_art': s['album_art'],
        'file': s['file']
    } for s in recommendations]
    return jsonify(rec_list)

@app.route('/graph')
def graph():
    """
    Create a 3D scatter plot of song embeddings using Plotly Express.
    Each point is annotated with the song title and artist.
    """
    data = []
    for song in songs_all:
        embedding = song['embedding']
        data.append({
            'id': song['id'],
            'title': song['title'],
            'artist': song['artist'],
            'e0': embedding[0],
            'e1': embedding[1],
            'e2': embedding[2]
        })
    df = pd.DataFrame(data)
    fig = px.scatter_3d(
        df,
        x='e0', y='e1', z='e2',
        hover_data=['title', 'artist'],
        title="Song Embeddings in 3D Space"
    )
    graph_html = fig.to_html(full_html=False)
    logging.info("Rendering 3D vector graph.")
    return render_template('graph.html', graph_html=graph_html)

@app.route('/data')
def data():
    """
    Endpoint to return current albums and individual songs as JSON.
    Used for client-side updates.
    """
    return jsonify({'albums': albums, 'songs': individual_songs})

if __name__ == '__main__':
    if not os.path.isdir(SONGS_FOLDER):
        os.makedirs(SONGS_FOLDER)
        logging.info(f"Created folder: {SONGS_FOLDER}. Place your audio files and album folders there.")
    logging.info("Starting Flask app.")
    socketio.run(app, debug=True)
