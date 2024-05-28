# Load Libraries
import os
import pandas as pd
import numpy as np
import time
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
# from decouple import config
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from wordcloud import WordCloud
from streamlit_searchbox import st_searchbox
from typing import Optional, Dict, Tuple, List, Union

# Page Config
st.set_page_config(page_title="Song-To-Song Comparison", 
                   page_icon="🆚")

from data_galaxy import init_spotify_client, get_spotify_data

# Class




# Initialize the Spotify client
sp = init_spotify_client()


###### Search Functions for Autocomplete Features ######
def artist_search_func(sp,query) -> List[str]:
    if not query:
        return []
    result = sp.search(q=query, type='artist', limit=5)
    artists = [artist['name'] for artist in result['artists']['items']]
    return artists
        
def track_search_func(sp,query) -> List[str]:
    if not query:
        return []
    result = sp.search(q=query, type='track', limit=10)
    tracks = [track['name'] for track in result['tracks']['items']]
    return tracks
    
def artist_track_search_func(sp, artist, query) -> List[str]:
    if not artist or not query:
        return []
    result = sp.search(q=f"artist:{artist} track:{query}", type='track', limit=10)
    tracks = [track['name'] for track in result['tracks']['items']]
    return tracks


#################################
#### application starts here ####
#################################

# Sidebar
with st.sidebar:
    st.title("Enter Your Track")
    # Artist1 input
    selected_artist1 = st_searchbox(label="Select Artist 1", 
                                   key="artist_input", 
                                   search_function=lambda query: artist_search_func(sp, query))
    # Track1 input
    selected_track1 = st_searchbox(label="Select Track 1", 
                                  key="track_input", 
                                  search_function=lambda query: artist_track_search_func(sp, selected_artist1, query))
   
    selected_artist2 = st_searchbox(label="Select Artist 2", 
                                   key="artist_input", 
                                   search_function=lambda query: artist_search_func(sp, query))
    
    selected_track2 = st_searchbox(label="Select Track 2", 
                                  key="track_input", 
                                  search_function=lambda query: artist_track_search_func(sp, selected_artist2, query))

    # Compare button
    compare_button = st.sidebar.button("Compare")
