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
from typing import Optional, Dict, Tuple, List, Union


# Page Config
st.set_page_config(page_title="Music Era Comparison", 
                   page_icon="📀")


class EraComparison:
    def __init__(self, sp) -> None:
        self.sp = sp
        self.playlists_ttl = {
        "All Out 50s": ("37i9dQZF1DWSV3Tk4GO2fq", 604800), # 1 week TTL
        "All Out 60s": ("37i9dQZF1DXaKIA8E7WcJj", 604800), 
        "All Out 70s": ("37i9dQZF1DWTJ7xPn4vNaz", 604800)
        }

    def retrieve_latest_data(self, _sp, playlist_id: str)-> Tuple[pd.Series, pd.DataFrame]:
        """ 

        """
        try:
            # Search TTL in the mapping
            ttl = None
            for _, (id, t) in self.playlists_ttl.items():
                if id == playlist_id:
                    ttl = t
                    break
            if ttl is None:
                st.error(f"Playlist ID '{playlist_id}' not found in the backend.")
                return pd.Series(), pd.DataFrame()

            # Inner function starts
            @st.cache_data(ttl=ttl, show_spinner=False)
            def _fetch_playlist_data(playlist_id:str) -> Tuple[pd.Series, pd.DataFrame]:
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
                        track = item['track']
                        artist_id = track['artists'][0]['id']
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
                    print(f"Error: {e}")
                    st.error("Failed to retrieve the latest Top 50 Tracks data. Please try again later.")
                    return pd.Series(), pd.DataFrame()
            return _fetch_playlist_data(playlist_id)
        
        except Exception as e:
            st.error(f"An error occured: {str(e)}")
            return pd.Series(), pd.DataFrame()


    def compare_playlists(self, playlist_id1:str, playlist_id2:str):
        att1, df1 = self.retrieve_latest_data(self.sp, playlist_id1)
        att2, df2 = self.retrieve_latest_data(self.sp, playlist_id2)
        if df1.empty or df2.empty:
            st.error("One of the playlists could not be retrieved successfully.")
            return
        return att1, df1, att2, df2
    

    def get_playlist_name(self, playlist_id:str):
        # Reverse the mappings in playlist1 and playlist2 so IDs are keys
        reversed_playlist1 = {v: k for k, v in playlist1.items()}
        reversed_playlist2 = {v: k for k, v in playlist2.items()}

        # Combine the reversed dictionaries
        playlist_map = {**reversed_playlist1, **reversed_playlist2}

        print("Playlist Map:", playlist_map) 
        print("Looking for ID:", playlist_id)

        return playlist_map.get(playlist_id, "Unknown Playlist")


    def radar_chart(self, att1, att2, labels=None):
        """ 
        Create a radar chart comparing selected attributes of two eras.
        
        Args:
            att1 (pd.Series): Selected attributes of the first era.
            att2 (pd.Series): Selected attributes of the second era.
            labels (list, optional): Labels for the eras being compared. Defaults to None.
            
        Returns:
            go.Figure: The radar chart comparing the two eras.
        """
        if labels is None:
            labels = ['Era 1', 'Era 2']
        
        # Assuming att1 and att2 are pandas Series with the same indices
        attributes = att1.index.tolist()

        # Convert attributes to percentages
        att1_values = (att1 * 100).tolist()
        att2_values = (att2 * 100).tolist()
        
        # Colors for the radar chart
        color_era1 = 'rgba(93, 58, 155, 0.9)'
        color_era2 = 'rgba(230, 97, 0, 0.7)'
        
        # Create the radar chart
        fig = go.Figure()
        
        # Trace for the first era
        fig.add_trace(go.Scatterpolar(
            r=att1_values,
            theta=attributes,
            fill='toself',
            name=labels[0],
            fillcolor=color_era1,
            line=dict(color=color_era1),
        ))
        
        # Trace for the second era
        fig.add_trace(go.Scatterpolar(
            r=att2_values,
            theta=attributes,
            fill='toself',
            name=labels[1],
            fillcolor=color_era2,
            line=dict(color=color_era2, dash='dot'),
        ))
        
        # Update layout
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            legend=dict(x=1, y=1, traceorder='normal', orientation='v', font=dict(color='black')),
            paper_bgcolor='lightgrey',
            font={"color": "black"},
            height=450,
            width=700,
            #title="title"
        )
        
        return fig
    
    def run_analysis(self, id1, id2):
        att1, df1, att2, df2 = self.compare_playlists(id1, id2)
        label1 = self.get_playlist_name(id1)  # Placeholder function
        label2 = self.get_playlist_name(id2) 
        
        st.header('Radar Chart Comparison:')
        st.text("Music Era Comparison of Attributes (Mean Values)")
        fig = self.radar_chart(att1, att2, labels=[label1, label2])
        st.plotly_chart(fig)


# Initialize the Spotify client
def init_spotify_client():
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]  #config('SPOTIFY_CLIENT_ID')
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"] # config('SPOTIFY_CLIENT_SECRET')
    credential_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=credential_manager)
    return sp
sp = init_spotify_client()


#################################
#### application starts here ####
#################################

playlist1 = {
    "Select a Playlist": None, # Placeholder value
    "All Out 50s": "37i9dQZF1DWSV3Tk4GO2fq", 
    "All Out 60s": "37i9dQZF1DXaKIA8E7WcJj", 
    "All Out 70s": "37i9dQZF1DWTJ7xPn4vNaz"
}

playlist2 = {
    "Select a Playlist": None, # Placeholder value
    "All Out 50s": "37i9dQZF1DWSV3Tk4GO2fq", 
    "All Out 60s": "37i9dQZF1DXaKIA8E7WcJj", 
    "All Out 70s": "37i9dQZF1DWTJ7xPn4vNaz"
}

with st.sidebar:
    st.title("Select Playlist By Era")

    # Playlist selection
    selected_playlist1 = st.selectbox("Select Era 1", options=list(playlist1.keys()), key='playlist1_selectbox')
    selected_playlist2 = st.selectbox("Select Era 2", options=list(playlist2.keys()), key='playlist2_selectbox')

    # Obtain playlist id
    playlist_id1 = playlist1[selected_playlist1]
    playlist_id2 = playlist2[selected_playlist2]

    # Compare button
    compare_button = st.sidebar.button("Compare")

# main
st.write("# Music Era Comparison")
# Header text
st.info("You're comparing...", icon="👩🏽‍🎤")

if compare_button and playlist_id1 and playlist_id2:
    try:
        era_comparision = EraComparison(sp)

        st.header('Compare 2 Different Era of Music')
        st.components.v1.iframe(f"https://open.spotify.com/embed/playlist/{playlist_id1}?utm_source=generator&theme=0",
                        width=500, height=160, scrolling=True)
        st.components.v1.iframe(f"https://open.spotify.com/embed/playlist/{playlist_id2}?utm_source=generator&theme=0",
                        width=500, height=160, scrolling=True)
        st.divider()
        
        # Run Analysis
        era_comparision.run_analysis(playlist_id1, playlist_id2)
        st.balloons()
    
    except Exception as e:
        print(e)
