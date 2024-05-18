# Load Libraries
import pandas as pd
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import plotly.graph_objects as go
import plotly.express as px
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
st.set_page_config(page_title="Top Chart Analysis", 
                   page_icon="👯🏼")

# Import data_galaxy after Page Config
from data_galaxy import init_spotify_client, fetch_artist_tracks

class SpotifyAnalyzer:
    def __init__(self, sp, artist_id: str) -> None:
        self.sp = sp
        self.mean_values_artist, self.df_artist = fetch_artist_tracks(self.sp, artist_id)
        self.colorscale=[  # Custom colorscale
            [0.0, "rgba(232, 148, 88, 0.9)"],   # Lighter orange
            [0.12, "rgba(213, 120, 98, 0.9)"],  # Dark orange
            [0.24, "rgba(190, 97, 111, 0.9)"],  # Reddish-pink
            [0.36, "rgba(164, 77, 126, 0.9)"],  # Lighter magenta
            [0.48, "rgba(136, 60, 137, 0.9)"],  # Deep magenta
            [0.58, "rgba(125, 50, 140, 0.9)"],  # Mid purple
            [0.68, "rgba(106, 44, 141, 0.9)"],  # Purple-pink
            [0.78, "rgba(87, 35, 142, 0.9)"],   # Deep purple
            [0.88, "rgba(69, 27, 140, 0.9)"],   # Rich purple
            [0.94, "rgba(40, 16, 137, 0.9)"],   # Darker purple
            [0.97, "rgba(26, 12, 135, 0.9)"],   # New shade between darker purple and dark blue
            [1.0, "rgba(12, 7, 134, 0.9)"]      # Dark blue
        ]


    def radar_chart(self) -> go.Figure:
        color_artist = "rgba(69, 27, 140, 0.9)" # Rich Purple
        mean_values_artist = self.mean_values_artist * 100
        att_list = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
        fig = go.Figure()

        # Add trace (mean)
        fig.add_trace(go.Scatterpolar(
            r=mean_values_artist,
            theta=att_list,
            fill='toself',
            name='Chart (mean)',
            fillcolor=color_artist,
            line=dict(color=color_artist),
        ))

        # Update the layout 
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            legend=dict(
                orientation='h',
                x=0.9,
                y=1.1,
            ),
            paper_bgcolor = 'WhiteSmoke',
            font = {"color": "black"},
            height=450,
            width=700,
            margin=dict(l=40, r=40, t=40, b=40),
            autosize=True
        )

        return fig
    

    def run_analysis(self) -> None:
        pass

# Initialize Spotify Client
sp = init_spotify_client()


def artist_search_func(sp,query) -> List[str]:
    result = sp.search(q=query, type='artist', limit=5)
    artists = [artist['name'] for artist in result['artists']['items']]
    return artists

# Function to get artist ID from artist name
def get_artist_id(sp, artist_name) -> str:
    result = sp.search(q=artist_name, type='artist', limit=1)
    if result['artists']['items']:
        return result['artists']['items'][0]['id']
    else:
        return None

with st.sidebar:
    st.title("Enter Artist Name")

    #placeholder
    selected_artist = None

    # Artist input
    selected_artist = st_searchbox(
        label="Select Artist", 
        key="artist_input", 
        search_function=lambda query: artist_search_func(sp, query),
        placeholder="Search for an artist..."
    )

    # If an artist is selected, fetch and display the artist ID
    if selected_artist:
        artist_id = get_artist_id(sp, selected_artist)
        st.write(f"Selected Artist: {selected_artist}")
        if artist_id:
            st.write(f"Artist ID: {artist_id}")
        else:
            st.write("Artist ID not found")

analyze_button = st.sidebar.button("Analyze")


#-------- Main ----------
st.markdown("# Artist Analysis")
st.info("Select an artist name to analyze. You'll get a comprehensive analysis of your selected album.", icon="📀")

try:
    if analyze_button:
        ...

except Exception as e:
    print(e)
    st.error(f'An error occurred: {str(e)}')