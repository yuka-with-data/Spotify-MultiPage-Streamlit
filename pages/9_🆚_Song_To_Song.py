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

from data_retriever import init_spotify_client, get_spotify_data

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
        
    def display_song_genres(self, track_data1, track_data2, label1, label2):
        for i, (track_data, label) in enumerate(zip([track_data1, track_data2], [label1, label2]), start=1):
            st.subheader(f"Genres for {label}")
            artist_genres = track_data['genres']
            if not artist_genres:
                st.warning("No genres found for this track.")
            else:
                # Split the genres string into individual genres
                individual_genres = artist_genres.split(',')
                # Create HTML for genres as badges with the specified background color
                badges_html = "".join([f"<span style='background-color:rgba(26, 12, 135, 0.9); color:#ffffff; padding:5px; border-radius:5px; margin:2px; display:inline-block;'>{genre.strip()}</span>" for genre in individual_genres])
                # Display the genres using HTML
                st.write(f"<div style='display: flex; flex-wrap: wrap; gap: 10px;'>{badges_html}</div>", unsafe_allow_html=True)


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
    
    def mode_comparison_emojis(self, track_data1, track_data2, label1, label2):
        mode1 = track_data1.get('mode', 0)
        mode2 = track_data2.get('mode', 0)
        
        emoji_major = "😊" if mode1 == 1 else "🙁"
        emoji_minor = "😊" if mode2 == 1 else "🙁"

        st.write(f"### {label1}: ", emoji_major)
        st.write(f"### {label2}: ", emoji_minor)

        # Add legend
        st.write("Legend:")
        st.write("😊- Major mode  🙁- Minor mode")

    
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
    
    def display_explicitness(self, track_data1, track_data2, label1, label2):
        
        # Define icons for explicit and non-explicit tracks
        explicit_icon = "🔞"  # You can use any appropriate explicit icon
        clean_icon = "🅲"      # You can use any appropriate clean icon
        
        # Determine explicitness for each track
        explicitness1 = "Explicit" if track_data1['is_explicit'] else "Clean"
        explicitness2 = "Explicit" if track_data2['is_explicit'] else "Clean"
        
        # Display explicitness using icons
        st.write(f"### {label1}: {explicit_icon if explicitness1 == 'Explicit' else clean_icon}")
        st.write(f"### {label2}: {explicit_icon if explicitness2 == 'Explicit' else clean_icon}")

        st.write("Legend")
        st.write("🔞: Explicit  🅲: Clean")
    
    def valence_danceability_interaction(self, track_data1, track_data2, label1, label2) -> go.Figure:
        # Extract values
        valence1 = track_data1.get('valence', 0)
        danceability1 = track_data1.get('danceability', 0)
        valence2 = track_data2.get('valence', 0)
        danceability2 = track_data2.get('danceability', 0)
        
        # Create a scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[valence1], y=[danceability1],
            mode='markers+text',
            marker=dict(color='rgba(89, 42, 154, 0.7)', size=12),
            #text=[label1],
            textposition='top center',
            name=label1
        ))
        
        fig.add_trace(go.Scatter(
            x=[valence2], y=[danceability2],
            mode='markers+text',
            marker=dict(color='rgba(230, 97, 0, 0.7)', size=12),
            #text=[label2],
            textposition='top center',
            name=label2
        ))
        
        fig.update_layout(
            xaxis_title='Valence',
            yaxis_title='Danceability',
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
    

    def energy_loudness_interaction(self, track_data1, track_data2, label1, label2) -> go.Figure:
        # Extract energy and loudness values
        energy1 = track_data1.get('energy', 0)
        loudness1 = track_data1.get('loudness', 0)
        energy2 = track_data2.get('energy', 0)
        loudness2 = track_data2.get('loudness', 0)
        
        # Create the scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[energy1], y=[loudness1],
            mode='markers+text',
            marker=dict(color='rgba(89, 42, 154, 0.7)', size=12),
            #text=[label1],
            textposition='top center',
            name=label1
        ))
        
        fig.add_trace(go.Scatter(
            x=[energy2], y=[loudness2],
            mode='markers+text',
            marker=dict(color='rgba(230, 97, 0, 0.7)', size=12),
            #text=[label2],
            textposition='top center',
            name=label2
        ))
        
        # Update layout
        fig.update_layout(
            xaxis_title='Energy',
            yaxis_title='Loudness (dB)',
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
    
    def acousticness_instrumentalness_interaction(self, track_data1, track_data2, label1, label2) -> go.Figure:
        # Extract acousticness and instrumentalness values
        acousticness1 = track_data1.get('acousticness', 0)
        instrumentalness1 = track_data1.get('instrumentalness', 0)
        acousticness2 = track_data2.get('acousticness', 0)
        instrumentalness2 = track_data2.get('instrumentalness', 0)
        
        # Create the scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[acousticness1], y=[instrumentalness1],
            mode='markers+text',
            marker=dict(color='rgba(89, 42, 154, 0.7)', size=12),
            #text=[label1],
            textposition='top center',
            name=label1
        ))
        
        fig.add_trace(go.Scatter(
            x=[acousticness2], y=[instrumentalness2],
            mode='markers+text',
            marker=dict(color='rgba(230, 97, 0, 0.7)', size=12),
            #text=[label2],
            textposition='top center',
            name=label2
        ))
        
        # Update layout
        fig.update_layout(
            xaxis_title='Acousticness',
            yaxis_title='Instrumentalness',
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
    
    
    def run_analysis(self, artist1:str, track1:str, artist2:str, track2:str):
        # This method is to run all the visualization method
        track_data1, _ = self.retrieve_track_data(artist1, track1)
        track_data2, _ = self.retrieve_track_data(artist2, track2)

        if track_data1 and track_data2:
            # Generate labels
            label1 = f"{artist1} - {track1}"
            label2 = f"{artist2} - {track2}"

            st.header("Genres")
            st.text("Comparison of Genres")
            self.display_song_genres(track_data1, track_data2, label1, label2)

            # Radar Chart
            st.header("Attribute Radar Chart")
            st.text("Track Attribute Comparison (Mean Values)")
            fig = self.radar_chart(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig, use_container_width=True)

            # Key Distribution
            st.header("Key Distribution")
            st.text("Distribution of Keys Comparison")
            fig = self.key_distribution(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig, use_container_width=True)

            st.header("Mode Comparison Emoji")
            st.text("Comparison of Modes (Major vs Minor)")
            self.mode_comparison_emojis(track_data1, track_data2, label1, label2)

            # Duration Bar Chart
            st.header("Duration Bar Chart")
            st.text("Duration Bar Chart Comparison")
            fig = self.duration_bar_chart(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig, use_container_width=True)

            # Tempo Bar Chart
            st.header("Tempo Bar Chart")
            st.text("Tempo Bar Chart Comparison")
            fig = self.tempo_bar_chart(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig, use_container_width=True)

            # Loudness Bar Chart
            st.header("Loudness Bar Chart")
            st.text("Loudness Bar Chart Comparison")
            fig = self.loudness_bar_chart(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig, use_container_width=True)

            # Track Popularity Gauge Chart
            st.header("Track Popularity Gauge Chart")
            st.text("Track Popularity Comparison")
            fig = self.track_popularity_gauge_chart(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig, use_container_width=True)

            st.header("Explicitness Status")
            st.text("Comparison of Explicitness Status")
            self.display_explicitness(track_data1, track_data2, label1, label2)

            # Valence Danceability Interaction
            st.header("Valence Danceability Interaction")
            st.text("Valence vs. Danceability Comparison")
            fig = self.valence_danceability_interaction(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig, use_container_width=True)

            # Energy Loudness Interaction
            st.header("Energy Loudness Interaction")
            st.text("Energy vs. Loudness Comparison")
            fig = self.energy_loudness_interaction(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig, use_container_width=True)

            # Acousticness Instrumentalness Interaction
            st.header("Acousticness Instrumentalness Interaction")
            st.text("Acousticness vs. Instrumentalness Comparison")
            fig = self.acousticness_instrumentalness_interaction(track_data1, track_data2, label1, label2)
            st.plotly_chart(fig, use_container_width=True)

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