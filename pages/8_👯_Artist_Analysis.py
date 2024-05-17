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
        self.mean_values_top_50, self.df_top_50 = fetch_artist_tracks(self.sp, artist_id)
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
