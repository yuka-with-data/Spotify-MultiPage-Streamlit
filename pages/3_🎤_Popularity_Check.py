# Load Libraries
import pandas as pd
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import plotly.graph_objects as go
import spotipy
import streamlit as st
from streamlit_searchbox import st_searchbox
from typing import Optional, Dict, Tuple, List, Union


# Page Config
st.set_page_config(page_title="Popularity Check", 
                   page_icon="🎙️")

class PopularityScore():
    def __init__(self, sp) -> None:
        self.sp=sp

    def get_track_id(self, artist_name: str, track_name: str) -> Optional[str]:
        """
        Search for a track by artist and track name, and return the Spotify track ID.
        Args:
          artist_name (str): The name of the artist.
          track_name (str): The name of the track.
        Returns:
          Optional[str]: The Spotify ID of the track, or None if not found.
        """
        query = f"artist:{artist_name} track:{track_name}"
        results = self.sp.search(q=query, type='track', limit=1)
        
        # Check if there are any tracks in the results
        items = results['tracks']['items']
        if items:
            # Return the first track's ID
            return items[0]['id']
        else:
            # Return None if no tracks were found
            return None
        
    def create_popularity_chart(self, track_id: str) -> go.Figure:
        """Create a popularity gauge chart displaying the current popularity score of a track
        Args:
        track_id (str): Spotify track ID
        Returns:
        go.Figure: Gauge chart
        """
        # Fetch track details using Spotify API
        track_details = self.sp.track(track_id)
        popularity = track_details.get('popularity', None)

        if popularity is not None:
            # Create a gauge chart
            fig = go.Figure()

            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=popularity,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Popularity Score"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': 'rgba(89, 42, 154, 1)'},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 100], 'color': 'lightgrey'},
                        {'range': [0, popularity], 'color': 'rgba(65, 105, 225, 0.5)'}
                    ],
                }
            ))

            # Background Color option: 
            # Snow, Honeydew, MintCream, LavenderBlush, GhostWhite, Seashell
            fig.update_layout(
                paper_bgcolor="LavenderBlush",
                font={"color": "black"},
                height=450,
                width=700,
                autosize=True
            )
        else:
            st.warning("Popularity data not available for this track.")

        return fig


# Initialize Spotify Client
def init_spotify_client():
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]  #config('SPOTIFY_CLIENT_ID')
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"] # config('SPOTIFY_CLIENT_SECRET')
    credential_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=credential_manager)
    return sp
sp = init_spotify_client()
# Instantiate class
popularity_score = PopularityScore(sp)

###### Search Functions for Autocomplete Features ######
def artist_search_func(sp,query) -> List[str]:
    result = sp.search(q=query, type='artist', limit=5)
    artists = [artist['name'] for artist in result['artists']['items']]
    return artists
    
def artist_track_search_func(sp, artist, query) -> List[str]:
    result = sp.search(q=f"artist:{artist} track:{query}", type='track', limit=10)
    tracks = [track['name'] for track in result['tracks']['items']]
    return tracks

# Sidebar
with st.sidebar:
    st.title("Enter Your Track")
    # Artist input
    selected_artist = st_searchbox(label="Select Artist", 
                                   key="artist_input", 
                                   search_function=lambda query: artist_search_func(sp, query))
    # Track input
    selected_track = st_searchbox(label="Select Track", 
                                  key="track_input", 
                                  search_function=lambda query: artist_track_search_func(sp, selected_artist, query))
    # Compare button
    analyze_button = st.sidebar.button("Analyze")

#------ Main --------
st.markdown("# Popularity Check")
st.info("Select an artist and track name. You'll check the latest Popularity Score of your track.", icon="📡")

try:
    if analyze_button:
        track_id = popularity_score.get_track_id(selected_artist, selected_track)
        if track_id:
            # User's Track Player
            st.header('Your Track')
            st.components.v1.iframe(f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator",
                                                width=500, height=160, scrolling=True)
            st.divider()

            st.subheader(f"Current Popularity Score for {selected_track} by {selected_artist}")
            pop_chart = popularity_score.create_popularity_chart(track_id)
            st.plotly_chart(pop_chart, use_container_width=True)
except Exception as e:
    print(e)
    st.error("An error occurred during analysis. Please try again.")
