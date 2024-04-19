# Load Libraries
import os
import pandas as pd
import numpy as np
import time
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import plotly.graph_objects as go
import streamlit as st
# from decouple import config
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from wordcloud import WordCloud
from streamlit_searchbox import st_searchbox
from typing import Optional, Dict, Tuple, List, Union, Any

# Page Config
st.set_page_config(page_title="Genre Analysis", 
                   page_icon="🌌")

@st.cache_data(ttl=604800, show_spinner=False)
def retrieve_tracks_by_genre(_sp, genre: str, limit: int = 50) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Retrieves and processes track data based on a specified genre.
    
    Args:
        _sp (spotipy.Spotify): Spotify API client
        genre (str): The genre to fetch tracks for
        limit (int): The number of tracks to fetch (maximum 50 for simplicity)
    
    Returns:
        A tuple containing two elements:
        - A pandas Series with the mean values of selected audio features
        - A pandas DataFrame containing detailed info about each track associated with the genre
    """
    try:
        search_query = f"genre:{genre}"
        result = _sp.search(q=search_query, type='track', limit=limit)
        tracks = result['tracks']['items']

        if not tracks:
            st.error('No tracks found for the specified genre')
            return pd.Series(), pd.DataFrame()

        # Initialize a progress bar in the app
        progress_bar = st.progress(0)

        tracks_data = []
        for track in tracks:
            artist_id = track['artists'][0]['id']
            track_id = track['id']
            genres = _sp.artist(artist_id)['genres']
            track_info = {
                'artist_name': track['artists'][0]['name'],
                'track_name': track['name'],
                'is_explicit': track['explicit'],
                'album_release_date': track['album']['release_date'],
                'genres': ', '.join(genres)  # Join genres list into a string
            }

            # Fetch audio features
            audio_features = _sp.audio_features(track_id)[0]
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
                'time_signature': audio_features['time_signature'],
                'popularity': track['popularity']
            })
            tracks_data.append(track_info)

        progress_bar.progress(100)
        success_placeholder = st.empty()
        success_placeholder.success(f"Retrieved {limit} tracks from the genre!")
        time.sleep(2)
        success_placeholder.empty()
        progress_bar.empty()

        df = pd.DataFrame(tracks_data)
        df = df.sort_values(by='popularity', ascending=False).head(limit)
        att_list = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
        selected_atts = df[att_list].mean()

        return selected_atts, df

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return pd.Series(), pd.DataFrame()
    

class GenreAnalyzer():
    def __init__(self, sp, selected_genre) -> None:
        self.sp = sp
        self.att, self.df_genre = retrieve_tracks_by_genre(self.sp, selected_genre)
        
    def run_analysis(self):
        st.header("Genre DataFrame")
        st.dataframe(self.df_genre)


# Initialize Spotify Client
def init_spotify_client():
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]  #config('SPOTIFY_CLIENT_ID')
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"] # config('SPOTIFY_CLIENT_SECRET')
    credential_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=credential_manager)
    return sp
sp = init_spotify_client()

@st.cache_data(ttl=604800, show_spinner=False)
def fetch_spotify_genres(_sp):
    """
    Fetch a list of available genres from Spotify's API.
    Args:
        sp (spotipy.Spotify): A spotipy Spotify client instance.
    Returns:
        List[str]: A list of genre strings supported by Spotify.
    """
    try:
        genres_response = _sp.recommendation_genre_seeds()
        return genres_response['genres']
    except Exception as e:
        st.error(f"An error occurred while fetching genres: {str(e)}")
        return []

# Example call to fetch genres
available_genres = fetch_spotify_genres(sp)
# print(available_genres)


def genre_search_func(query:str, available_genres: List[str], limit: int = 5) -> List[str]:
    """
    Search for a genre in the list of available genres.
    Args:
        query (str): The query string to search for.
        limit (int): The maximum number of results to return.
    Returns:
        List[str]: A list of matching genres.
    """
    query = query.lower()
    # Filter genres that start with the query and limit the results
    filtered_genres = [genre for genre in available_genres if genre.lower().startswith(query)][:limit]
    return filtered_genres[:limit]


with st.sidebar:
    st.title("Select A Genre:")
    selected_genre = st_searchbox(label="Search Available Genre",
                                  key="genre_input",
                                  search_function=lambda query: genre_search_func(query, available_genres))
    analyze_button = st.sidebar.button("Analyze")   

# Main
st.write("# Genre Analysis")

try:
    if analyze_button and selected_genre:
        genre_analyzer = GenreAnalyzer(sp, selected_genre)
        genre_analyzer.run_analysis()

except Exception as e:
    print(e)
    st.error(f'An error occurred: {str(e)}')



