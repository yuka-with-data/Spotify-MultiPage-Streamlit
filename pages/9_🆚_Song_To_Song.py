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
class SpotifyAnalyzer:
    def __init__(self, sp) -> None:
        self.sp = sp
        # fetch track data
    
    def retrieve_track_data(self, artist, track):
        try:
            audio_features, track_id = get_spotify_data(self.sp, artist, track) # add placeholder for None
            return audio_features, track_id
        except Exception as e:
            print(f"Error retrieving track data: {e}")
            return None, None
        

    def radar_chart(self, track_data1, track_data2, label1, label2):
        attributes = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']

        # extract values for track 1 and 2
        att1_values = [(track_data1.get(attr, 0) * 100) for attr in attributes]
        att2_values = [(track_data2.get(attr, 0) * 100) for attr in attributes]

        # Colors for the radar chart
        color_track1 = "rgba(69, 27, 140, 0.9)"  # Rich Purple
        color_track2 = 'rgba(230, 97, 0, 0.7)'  # Rich Orange
        
        # Create the radar chart
        fig = go.Figure()

        # Trace for the first track
        fig.add_trace(go.Scatterpolar(
            r=att1_values,
            theta=attributes,
            fill='toself',
            name=label1,
            fillcolor=color_track1,
            line=dict(color=color_track1),
        ))
        
        # Trace for the second track
        fig.add_trace(go.Scatterpolar(
            r=att2_values,
            theta=attributes,
            fill='toself',
            name=label2,
            fillcolor=color_track2,
            line=dict(color=color_track2, dash='dot'),
        ))

        # Update layout
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            legend=dict(
                orientation='h',
                x=1,
                y=1.1,
                xanchor='right',
                yanchor='top'
            ),
            paper_bgcolor='WhiteSmoke',
            font={"color": "black"},
            margin=dict(l=40, r=40, t=55, b=40),
            height=450,
            width=700,
            autosize=True
        )
        
        return fig
    
    def key_distribution(self, track_data1, track_data2, label1, label2):
        keys = {
            0: 'C', 
            1: 'C#/Db', 
            2: 'D', 
            3: 'D#/Eb', 
            4: 'E', 
            5: 'F', 
            6: 'F#/Gb', 
            7: 'G', 
            8: 'G#/Ab', 
            9: 'A', 
            10: 'A#/Bb', 
            11: 'B'
        }

        key_data = [track_data1['key'], track_data2['key']]
        key_labels = [keys[key] for key in key_data]

        if key_data[0] == key_data[1]:
            y_values = [1, 1.1]  # Apply a small offset to one of the labels
        else:
            y_values = [1, 1]

        fig = go.Figure(data=[
            go.Scatter(
                x=key_labels, 
                y=y_values, 
                mode='markers+text', 
                marker=dict(color=['rgba(89, 42, 154, 0.7)', 'rgba(230, 97, 0, 0.7)'], size=20),
                text=[label1, label2],
                textposition='top center'
            )
        ])

        fig.update_layout(
            xaxis_title='Key',
            yaxis=dict(
                visible=False,
                showticklabels=False
            ),
            showlegend=False,
            paper_bgcolor='WhiteSmoke',
            template='plotly_white',
            font={'color': "black"},
            height=450,
            width=700, 
            margin=dict(l=50, r=50, t=50, b=50),
            autosize=True
        )

        return fig
    
    def duration_bar_chart(self, track_data1, track_data2, label1, label2) -> go.Figure:
        # Extract duration values in milliseconds from track data and convert to minutes
        duration1 = track_data1.get('duration_ms', 0) / 60000  # convert ms to minutes
        duration2 = track_data2.get('duration_ms', 0) / 60000  # convert ms to minutes

        # Create a bar chart
        fig = go.Figure(data=[
            go.Bar(name=label1, x=["Duration"], y=[duration1], marker_color='rgba(89, 42, 154, 0.7)'),
            go.Bar(name=label2, x=["Duration"], y=[duration2], marker_color='rgba(230, 97, 0, 0.7)')
        ])
        
        fig.update_layout(
            yaxis_title='Duration (Minutes)',
            barmode='group',
            paper_bgcolor='WhiteSmoke',
            template='plotly_white',
            font={'color': "black"},
            legend=dict(
                orientation='h',
                x=1,
                y=1.1,
                xanchor='right',
                yanchor='top'
            ),
            margin=dict(l=40, r=40, t=55, b=40),
            height=450,
            width=700,
            autosize=True
        )
        
        return fig

    
    def tempo_bar_chart(self, track_data1, track_data2, label1, label2) -> go.Figure:
        # Extract tempo values from track data
        tempo1 = track_data1.get('tempo', 0)
        tempo2 = track_data2.get('tempo', 0)
        
        # Create a bar chart
        fig = go.Figure(data=[
            go.Bar(name=label1, x=["Tempo"], y=[tempo1], marker_color='rgba(89, 42, 154, 0.7)'),
            go.Bar(name=label2, x=["Tempo"], y=[tempo2], marker_color='rgba(230, 97, 0, 0.7)')
        ])
        
        fig.update_layout(
            yaxis_title='Tempo (BPM)',
            barmode='group',
            paper_bgcolor='WhiteSmoke',
            template='plotly_white',
            font={'color': "black"},
            legend=dict(
                orientation='h',
                x=1,
                y=1.1,
                xanchor='right',
                yanchor='top'
            ),
            margin=dict(l=40, r=40, t=55, b=40),
            height=450,
            width=700,
            autosize=True
        )
        
        return fig

    
    def loudness_bar_chart(self, track_data1, track_data2, label1, label2) -> go.Figure:
        # Extract loudness values from track data
        loudness1 = track_data1.get('loudness', 0)
        loudness2 = track_data2.get('loudness', 0)
        
        # Create a bar chart
        fig = go.Figure(data=[
            go.Bar(name=label1, x=["Loudness"], y=[loudness1], marker_color='rgba(89, 42, 154, 0.7)'),
            go.Bar(name=label2, x=["Loudness"], y=[loudness2], marker_color='rgba(230, 97, 0, 0.7)')
        ])
        
        fig.update_layout(
            yaxis_title='Loudness (dB)',
            barmode='group',
            paper_bgcolor='WhiteSmoke',
            template='plotly_white',
            font={'color': "black"},
            legend=dict(
                orientation='h',
                x=1,
                y=1.1,
                xanchor='right',
                yanchor='top'
            ),
            margin=dict(l=40, r=40, t=55, b=40),
            height=450,
            width=700,
            autosize=True
        )
        
        return fig

    
    def track_popularity_gauge_chart(self, track_data1, track_data2, label1, label2) -> go.Figure:
        # Extract popularity scores from track data
        popularity1 = track_data1.get('popularity', 0)
        popularity2 = track_data2.get('popularity', 0)

        # Create a gauge chart for track 1
        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=popularity1,
            domain={'x': [0, 0.45], 'y': [0, 1]},
            title={'text': label1},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': 'rgba(89, 42, 154, 1)'},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 100], 'color': 'GhostWhite'},
                    {'range': [0, popularity1], 'color': 'rgba(26, 12, 135, 0.5)'}
                ],
            }
        ))

        # Create a gauge chart for track 2
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=popularity2,
            domain={'x': [0.55, 1], 'y': [0, 1]},
            title={'text': label2},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': 'rgba(230, 97, 0, 1)'},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 100], 'color': 'GhostWhite'},
                    {'range': [0, popularity2], 'color': 'rgba(230, 97, 0, 0.5)'}
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor='WhiteSmoke',
            template='plotly_white',
            font={'color': "black"},
            height=450,
            width=700,
            autosize=True
        )

        return fig
    
    
    def run_analysis(self, artist1:str, track1:str, artist2:str, track2:str):
        # This method is to run all the visualization method
        track_data1, _ = self.retrieve_track_data(artist1, track1)
        track_data2, _ = self.retrieve_track_data(artist2, track2)

        if track_data1 and track_data2:
            # Generate labels
            label1 = f"{artist1} - {track1}"
            label2 = f"{artist2} - {track2}"

            # Radar Chart
            st.header("Radar Chart Comparison")
            st.text("Track Attribute Comparison (Mean Values)")
            fig = self.radar_chart(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig)

            # Key Distribution
            st.header("Key Distribution")
            st.text("Distribution of Keys Comparison")
            fig = self.key_distribution(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig)

            # Duration Bar Chart
            st.header("Duration Bar Chart")
            st.text("Duration Bar Chart Comparison")
            fig = self.duration_bar_chart(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig)

            # Tempo Bar Chart
            st.header("Tempo Bar Chart")
            st.text("Tempo Bar Chart Comparison")
            fig = self.tempo_bar_chart(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig)

            # Loudness Bar Chart
            st.header("Loudness Bar Chart")
            st.text("Loudness Bar Chart Comparison")
            fig = self.loudness_bar_chart(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig)

            # Track Popularity Gauge Chart
            st.header("Track Popularity Gauge Chart")
            st.text("Track Popularity Comparison")
            fig = self.track_popularity_gauge_chart(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig)
    
        else:
            st.error("Error retrieving track data for one or both tracks.")
       


    
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
    st.title("Enter 2 Tracks")
    # Artist1 input
    selected_artist1 = st_searchbox(label="Select Artist 1", 
                                   key="artist1_input", 
                                   search_function=lambda query: artist_search_func(sp, query))
    # Track1 input
    selected_track1 = st_searchbox(label="Select Track 1", 
                                  key="track1_input", 
                                  search_function=lambda query: artist_track_search_func(sp, selected_artist1, query))
   
    selected_artist2 = st_searchbox(label="Select Artist 2", 
                                   key="artist2_input", 
                                   search_function=lambda query: artist_search_func(sp, query))
    
    selected_track2 = st_searchbox(label="Select Track 2", 
                                  key="track2_input", 
                                  search_function=lambda query: artist_track_search_func(sp, selected_artist2, query))

    # Compare button
    compare_button = st.sidebar.button("Compare")

st.write("## Song-To-Song Comparison")
st.info("Select two tracks and compare their similarity and differences in their music data.", icon='✨')

# Main section
if compare_button and selected_track1 and selected_track2:
    try:
        track_comparison = SpotifyAnalyzer(sp)

        _, id1 = track_comparison.retrieve_track_data(selected_artist1, selected_track1)
        _, id2 = track_comparison.retrieve_track_data(selected_artist2, selected_track2)

        st.subheader("Track 1")
        st.components.v1.iframe(
            f"https://open.spotify.com/embed/track/{id1}?utm_source=generator&theme=0",
            width=500,
            height=160,
            scrolling=True
        )

        st.subheader("Track 2")
        st.components.v1.iframe(
            f"https://open.spotify.com/embed/track/{id2}?utm_source=generator&theme=0",
            width=500,
            height=160,
            scrolling=True
        )

        # Run analysis
        track_comparison.run_analysis(selected_artist1, selected_track1, selected_artist2, selected_track2)
        st.balloons()
    
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("Please select two tracks to compare.")