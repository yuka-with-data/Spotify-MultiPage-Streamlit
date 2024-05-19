import time
import pandas as pd
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import Optional, Dict, Tuple, List, Union

# Initialize the Spotify client
@st.cache_data(ttl=86400)
def init_spotify_client():
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]  #config('SPOTIFY_CLIENT_ID')
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"] # config('SPOTIFY_CLIENT_SECRET')
    credential_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=credential_manager)
    return sp
sp = init_spotify_client()


# Playlist data retrieval
def retrieve_playlist_data(_sp, playlist_id: str)-> Tuple[pd.Series, pd.DataFrame]:
    try:

        @st.cache_data(ttl=86400, show_spinner=False)
        def fetch_playlist_data(playlist_id:str) -> Tuple[pd.Series, pd.DataFrame]:
            try:
                print(f"Fetching data for playlist ID: {playlist_id}")
                tracks_data = []
                results = _sp.playlist_tracks(playlist_id)

                # When no tracks found
                total_tracks = len(results['items'])
                if total_tracks == 0:
                    st.error('No tracks found in the playlist')
                    return pd.Series(), pd.DataFrame()
                
                # Initialize a progress bar in the app
                progress_bar = st.progress(0)

                for index, item in enumerate(results['items']):
                    # Update progress bar based on the number of tracks proceeded
                    percent_complete = int((index + 1) / total_tracks * 100)
                    progress_bar.progress(percent_complete, text="🛰️Fetching The Most Up-To-Date Chart Data. Please Wait.")

                    # Track
                    track = item['track']
                    if not track or not track.get('id'):
                        continue # Skip tracks with missing info
                    
                    # Artist
                    artist = track['artists'][0] if track['artists'] else None
                    if not artist:
                        continue # Skip tracks with missing artist info
                    
                    # Track Info
                    track_id = track['id']
                    genres = _sp.artist(artist['id'])['genres']

                    track_info = {
                        'artist_name': artist['name'],
                        'track_name': track['name'],
                        'is_explicit': track['explicit'],
                        'album_release_date': track['album']['release_date'],
                        'genres': ', '.join(genres)  # Join genres list into a string
                    }

                    # Fetch audio features
                    audio_features = _sp.audio_features(track_id)[0]
                    if not audio_features:
                        continue # Skip tracks with missing audio features
                    track_info.update({
                        'danceability': audio_features['danceability'],
                        'valence': audio_features['valence'],
                        'energy': audio_features['energy'],
                        'loudness': audio_features['loudness'],
                        'acousticness': audio_features['acousticness'],
                        'instrumentalness': audio_features['instrumentalness'],
                        'liveness': audio_features['liveness'],
                        'speechiness': audio_features['speechiness'],
                        'key': audio_features['key'],
                        'tempo': audio_features['tempo'],
                        'mode': audio_features['mode'],
                        'duration_ms': audio_features['duration_ms'],
                        'time_signature': audio_features['time_signature']
                    })
                    # Fetch Popularity
                    search_query = f"{track['name']} {artist['name']}"
                    popularity_result = _sp.search(q=search_query, type='track', limit=1)
                    print(f"Popularity Result: {popularity_result}")
                    if popularity_result['tracks']['items']:
                        popularity = popularity_result['tracks']['items'][0]['popularity']
                    else:
                        popularity = None
                    track_info['popularity'] = popularity

                    tracks_data.append(track_info)
                
                # Progress bar complete
                progress_bar.progress(100)
                # Success msg with a placeholder
                success_placeholder = st.empty()
                success_placeholder.success(f"Retrieved {total_tracks} Top Tracks from the playlist!", icon="✅")
                # Display the msg for 2 seconds
                time.sleep(2)
                
                success_placeholder.empty()
                progress_bar.empty()

                # Save the tracks data to a DataFrame
                df = pd.DataFrame(tracks_data)
                # Calculate mean values for each attribute
                att_list = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
                selected_atts = df[att_list].mean()

                return selected_atts, df

            except Exception as e:
                print(f"Error 1: {e}")
                st.error("Failed to retrieve the latest Top 50 Tracks data. Please try again later.")
                return pd.Series(), pd.DataFrame()
        return fetch_playlist_data(playlist_id)
    
    except Exception as e:
        print(f"Error 2: {e}")
        st.error(f"An error occured: {str(e)}")
        return pd.Series(), pd.DataFrame()
    

# Album Data Retrieval
def retrieve_album_data(_sp, album_id: str) -> Tuple[pd.Series, pd.DataFrame]:
    try:
        @st.cache_data(ttl=86400, show_spinner=False)
        def fetch_album_data(album_id: str) -> Tuple[pd.Series, pd.DataFrame]:
            try:
                tracks_data = []
                results = _sp.album_tracks(album_id)
                if not results or 'items' not in results or len(results['items']) == 0:
                    st.error('No tracks found in the album')
                    return pd.Series(), pd.DataFrame()
                
                # Fetch album's details 
                album_details = _sp.album(album_id) if album_id else None
                album_release = album_details.get('release_date', 'Unknown')
                album_genres = ', '.join(album_details.get('genres', []))

                # Initialize a progress bar in the app
                progress_bar = st.progress(0)
                total_tracks = len(results['items'])

                for index, track in enumerate(results['items']):
                    track_id = track.get('id')
                    if not track_id:
                        continue # skip tracks with missing info
                    
                    # Fetch detailed track information to get popularity
                    detailed_track_info = _sp.track(track_id)
                    popularity = detailed_track_info['popularity']

                    artist_id = track['artists'][0]['id'] if track['artists'] else None
                    genres = _sp.artist(artist_id)['genres'] if artist_id else None

                    track_info = {
                        'artist_name': track['artists'][0]['name'],
                        'track_name': track['name'],
                        'is_explicit': track['explicit'],
                        'album_release_date': album_release,
                        'artist_genres': ', '.join(genres),
                        'album_genres': album_genres,
                        'popularity': popularity
                    }

                    # Fetch audio features
                    audio_features = _sp.audio_features(track_id)[0] if track_id else None
                    if audio_features:
                        track_info.update({
                            'danceability': audio_features.get('danceability', 0),
                            'valence': audio_features.get('valence', 0),
                            'energy': audio_features.get('energy', 0),
                            'loudness': audio_features.get('loudness', 0),
                            'acousticness': audio_features.get('acousticness', 0),
                            'instrumentalness': audio_features.get('instrumentalness', 0),
                            'liveness': audio_features.get('liveness', 0),
                            'speechiness': audio_features.get('speechiness', 0),
                            'key': audio_features.get('key', 0),
                            'tempo': audio_features.get('tempo', 0),
                            'mode': audio_features.get('mode', 0),
                            'duration_ms': audio_features.get('duration_ms', 0),
                            'time_signature': audio_features.get('time_signature', 0)
                        })
                    else:
                        st.error(f"Failed to retrieve audio features for track {track_id}")

                    tracks_data.append(track_info)
                    # Update progress bar
                    percent_complete = int((index + 1) / total_tracks * 100)
                    progress_bar.progress(percent_complete)

                # Progress bar and success message cleanup
                progress_bar.progress(100)
                # Success msg with a placeholder
                success_placeholder = st.empty()
                success_placeholder.success(f"Retrieved {total_tracks} tracks from the album!")
                # Display the msg for 2 seconds
                time.sleep(2)
                success_placeholder.empty()
                progress_bar.empty()
                
                # Save the tracks data to a DataFrame
                df = pd.DataFrame(tracks_data)
                # Calculate mean values for each audio attribute
                audio_features_keys = ['danceability', 
                                        'valence', 
                                        'energy', 
                                        'loudness', 
                                        'acousticness', 
                                        'instrumentalness', 
                                        'liveness', 
                                        'speechiness', 
                                        'tempo']
                selected_atts = df[audio_features_keys].mean()
                return selected_atts, df
            
            except Exception as e:
                print(f"Error 3: {e}")
                st.error("Failed to retrieve the latest Top 50 Tracks data. Please try again later.")
                return pd.Series(), pd.DataFrame()
        
        return fetch_album_data(album_id)
    
    except Exception as e:
        print(f"Error 4: {e}")
        st.error(f"An error occured: {str(e)}")
        return pd.Series(), pd.DataFrame()
        

""" 
 Era data retriever is slightly different from others, 
 since the era data has to be generated twice, actual retrieval will be done
 inside of Class. The data caching will be done in this script.
 """
@st.cache_data(ttl=604800, show_spinner=False)
def retrieve_era_data(_sp, playlist_id:str) -> Tuple[pd.Series, pd.DataFrame]:
    try:
        tracks_data = []
        results = _sp.playlist_tracks(playlist_id)

        # When no tracks found
        total_tracks = len(results['items'])
        if total_tracks == 0:
            st.error('No tracks found in the playlist')
            return pd.Series(), pd.DataFrame()
        
        # Initialize a progress bar in the app
        progress_bar = st.progress(0)

        for index, item in enumerate(results['items']):
            # Update progress bar based on the number of tracks proceeded
            percent_complete = int((index + 1) / total_tracks * 100)
            progress_bar.progress(percent_complete, text="🛰️Fetching The Most Up-To-Date Chart Data. Please Wait.")

            # Track
            track = item.get('track')
            if not track or not track.get('id'):
                continue # skip tracks with missing info

            # Artist
            artist = track['artists'][0] if track['artists'] else None
            if not artist:
                continue # skip tracks with missing artist info
            
            # Track info
            track_id = track['id']
            genres = _sp.artist(track['artists'][0]['id'])['genres']

            track_info = {
                'artist_name': track['artists'][0]['name'],
                'track_name': track['name'],
                'is_explicit': track['explicit'],
                'album_release_date': track['album']['release_date'],
                'genres': ', '.join(genres)  # Join genres list into a string
            }

            # Fetch audio features
            audio_features = _sp.audio_features(track_id)[0]
            if not audio_features:
                continue # skip tracks with missing audio features
            track_info.update({
                'danceability': audio_features['danceability'],
                'valence': audio_features['valence'],
                'energy': audio_features['energy'],
                'loudness': audio_features['loudness'],
                'acousticness': audio_features['acousticness'],
                'instrumentalness': audio_features['instrumentalness'],
                'liveness': audio_features['liveness'],
                'speechiness': audio_features['speechiness'],
                'key': audio_features['key'],
                'tempo': audio_features['tempo'],
                'mode': audio_features['mode'],
                'duration_ms': audio_features['duration_ms'],
                'time_signature': audio_features['time_signature']
            })
            # Fetch Popularity
            search_query = f"{track['name']} {track['artists'][0]['name']}"
            popularity_result = _sp.search(q=search_query, type='track', limit=1)
            print(f"Popularity Result: {popularity_result}")
            if popularity_result['tracks']['items']:
                popularity = popularity_result['tracks']['items'][0]['popularity']
            else:
                popularity = None
            track_info['popularity'] = popularity

            tracks_data.append(track_info)
        
        # Progress bar complete
        progress_bar.progress(100)
        # Success msg with a placeholder
        success_placeholder = st.empty()
        success_placeholder.success(f"Retrieved {total_tracks} Top Tracks from the playlist!", icon="✅")
        # Display the msg for 2 seconds
        time.sleep(2)
        
        success_placeholder.empty()
        progress_bar.empty()

        # Save the tracks data to a DataFrame
        df = pd.DataFrame(tracks_data)
        # Calculate mean values for each attribute
        att_list = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
        # Calculate mean of attributes
        selected_atts = df[att_list].mean()
        return selected_atts, df

    except Exception as e:
        print(f"Error 5: {e}")
        st.error("Failed to retrieve the latest Top 50 Tracks data. Please try again later.")
        return pd.Series(), pd.DataFrame()
    
# Artist Data Retrieval
@st.cache_data(ttl=604800, show_spinner=False)
def fetch_artist_tracks(_sp, artist_id):
    tracks_data = []
    albums = []

    # Fetch all albums from the artist
    results = _sp.artist_albums(artist_id, album_type='album,single')
    albums.extend(results['items'])
    # Albums pagination
    while results['next']:
        results = _sp.next(results)
        albums.extend(results['items'])

    # Fetch genres
    artist_genres = _sp.artist(artist_id)['genres']

    # Iterate through each album to fetch tracks
    for album in albums:
        results = _sp.album_tracks(album['id'])
        tracks = results['items']
        # Tracks pagination
        while results['next']:
            results = _sp.next(results)
            tracks.extend(results['items'])

        for track in tracks:
            # Only proceed if 'track' is not None and the first artist matches the artist_id
            if track is not None and track['artists'][0]['id'] == artist_id:
                track_info = {
                    'artist_name': ', '.join([artist['name'] for artist in track['artists']]),  # Handling multiple artists
                    'track_name': track['name'],
                    'album_name': album['name'],
                    'is_explicit': track['explicit'],
                    'genres': ', '.join(artist_genres),
                    'release_date': album['release_date'],
                    'track_id': track['id'],
                }

                # Fetch audio features and update track_info if needed
                audio_features = _sp.audio_features(track['id'])[0]
                if audio_features:
                    track_info.update({
                        'danceability': audio_features['danceability'],
                        'valence': audio_features['valence'],
                        'energy': audio_features['energy'],
                        'loudness': audio_features['loudness'],
                        'acousticness': audio_features['acousticness'],
                        'instrumentalness': audio_features['instrumentalness'],
                        'liveness': audio_features['liveness'],
                        'speechiness': audio_features['speechiness'],
                        'key': audio_features['key'],
                        'tempo': audio_features['tempo'],
                        'mode': audio_features['mode'],
                        'duration_ms': audio_features['duration_ms'],
                        'time_signature': audio_features['time_signature']
                    })

                tracks_data.append(track_info)
    
    df = pd.DataFrame(tracks_data)

    # Calculate mean values for each attribute
    att_list = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
    # Calculate mean of attributes
    selected_atts = df[att_list].mean()

    return selected_atts, df