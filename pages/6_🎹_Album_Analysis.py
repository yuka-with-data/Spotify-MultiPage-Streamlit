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
import warnings

# Suppress specific seaborn warning
warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue` is deprecated")


# Page Config
st.set_page_config(page_title="Album Analysis", 
                   page_icon="🎹")

# Retrieve album data
def retrieve_album_data(_sp, album_id:str) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Retrieves and processes the latest tracks data from a Spotify album.
    Args:
       _sp (spotipy.Spotify): Spotify API client
       album_id (str): Spotify album id.
    Returns:
       A tuple containing two elements:
       - A pandas Series with the mean values of selected audio features
       - A pandas DataFrame containing detailed info about each track in the album
    """
    try:
        # Inner function starts
        @st.cache_data(show_spinner=False)
        def _fetch_album_data(album_id: str) -> Tuple[pd.Series, pd.DataFrame]:
            try:
                tracks_data = []
                results = _sp.album_tracks(album_id)
                total_tracks = len(results['items'])

                if total_tracks == 0:
                    st.error('No tracks found in the album')
                    return pd.Series(), pd.DataFrame()
                
                # Fetch album's details 
                album_details = _sp.album(album_id)
                album_release = album_details['release_date']
                album_genres = ', '.join(album_details['genres'])

                # Initialize a progress bar in the app
                progress_bar = st.progress(0)

                for index, track in enumerate(results['items']):
                    # Update progress bar based on the number of tracks proceeded
                    percent_complete = int((index + 1) / total_tracks * 100)
                    progress_bar.progress(percent_complete)

                    artist_id = track['artists'][0]['id']
                    track_id = track['id']

                    # Fetch detailed track information to get popularity
                    detailed_track_info = _sp.track(track_id)
                    popularity = detailed_track_info['popularity']

                    genres = _sp.artist(artist_id)['genres']

                    track_info = {
                        'artist_name': track['artists'][0]['name'],
                        'track_name': track['name'],
                        'is_explicit': track['explicit'],
                        'album_release_date': album_release,
                        'artist_genres': ', '.join(genres),
                        'album_genres': album_genres
                    }

                    # Fetch audio features
                    audio_features = _sp.audio_features(track_id)[0]
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

                    # Fetch popularity
                    popularity = popularity
                    track_info['popularity'] = popularity

                    tracks_data.append(track_info)

                # Progress bar and success message cleanup
                progress_bar.progress(100)
                st.success(f"Retrieved {total_tracks} tracks from the album!")

                # Save the tracks data to a DataFrame
                df = pd.DataFrame(tracks_data)
                # print(df)
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
                print(f"Error: {e}")
                st.error("Failed to retrieve album data. Please try again later.")
                return pd.Series(), pd.DataFrame()
            
        return _fetch_album_data(album_id)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return pd.Series(), pd.DataFrame()
    

# Class
class AlbumAnalyzer:
    def __init__(self, sp, album_id: str) -> None:
        self.sp = sp
        self.mean_values_album, self.df_album = retrieve_album_data(self.sp, album_id)

    def radar_chart(self) -> go.Figure:
        color_album = 'rgba(255, 99, 132, 0.9)'
        audio_features_keys = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']

        filtered_values_album_percent = self.mean_values_album[audio_features_keys]
        mean_values_album_percent = filtered_values_album_percent.values * 100
        print(f"mean value album: {mean_values_album_percent}")

        fig = go.Figure()

        # Add trace (mean)
        fig.add_trace(go.Scatterpolar(
            r=mean_values_album_percent,
            theta=audio_features_keys,
            fill='toself',
            name='Album (mean)',
            fillcolor=color_album,
            line=dict(color=color_album),
        ))

        # Update the layout
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            paper_bgcolor='white',
            font={"color": "black"},
            height=450,
            width=700
        )

        return fig
    
    def tempo_histogram(self) -> plt.Figure:
        color_album = cm.plasma(0.15)
        color_average_tempo = cm.plasma(0.55) 

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df_album['tempo'],
                     bins=50,
                     # kde=True,
                     color=color_album,
                     edgecolor='black',
                     ax=ax)
        mean_tempo = self.df_album['tempo'].mean()
        ax.axvline(mean_tempo,
                   color=color_average_tempo,
                   linestyle='dashed',
                   linewidth=2,
                   label='Average Tempo')
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Frequency')
        ax.legend()

        max_count = int(max(ax.get_yticks())) # Find the current max y-tick and round up
        ax.set_yticks(range(0, max_count))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # Set y-ticks to integers

        ax.grid(False, axis='x')
        ax.grid(True, axis='y', linestyle='--',alpha=0.6)

        fig.patch.set_facecolor('lightgrey')
        ax.set_facecolor('lightgrey')

        return fig

    def duration_histogram(self) -> plt.Figure:
        color_album = cm.plasma(0.15)
        color_average_duration = cm.plasma(0.55)

        self.df_album['duration_min'] = self.df_album['duration_ms'] / 60000 # Convert duration from milliseconds to minutes for readability

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df_album['duration_min'],
                     bins=50,
                     # kde=True,
                     color=color_album,
                     edgecolor='black',
                     ax=ax)
        mean_duration = self.df_album['duration_min'].mean()
        ax.axvline(mean_duration,
                   color=color_average_duration,
                   linestyle='dashed',
                   linewidth=2,
                   label='Average Duration')
        ax.set_xlabel('Duration (min)')
        ax.set_ylabel('Frequency')
        ax.legend()

        max_count = int(max(ax.get_yticks())) # Find the current max y-tick and round up
        ax.set_yticks(range(0, max_count))

        ax.grid(False, axis='x')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        fig.patch.set_facecolor('lightgrey')
        ax.set_facecolor('lightgrey')

        return fig
    
    def loudness_histogram(self) -> plt.Figure:
        color_album = cm.plasma(0.15)
        color_average_loud = cm.plasma(0.55)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df_album['loudness'],
                     bins=50,
                     # kde=True,
                     color=color_album,
                     edgecolor='black',
                     ax=ax)
        mean_loudness = self.df_album['loudness'].mean()
        ax.axvline(mean_loudness,
                   color=color_average_loud,
                   linestyle='dashed',
                   linewidth=2,
                   label='Average Loudness')
        ax.set_xlabel('Loudness')
        ax.set_ylabel('Frequency')
        ax.legend()

        max_count = int(max(ax.get_yticks())) # Find the current max y-tick and round up
        ax.set_yticks(range(0, max_count))

        ax.grid(False, axis='x')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        fig.patch.set_facecolor('lightgrey')
        ax.set_facecolor('lightgrey')

        return fig
    
    
    def key_histogram(self) -> plt.Figure:
        color_album = cm.plasma(0.15)
        color_average_key = cm.plasma(0.55)

        # Mapping of numeric key values to corresponding alphabetic keys
        key_mapping = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

        # Get key counts and ensure all keys are present
        key_counts = self.df_album['key'].value_counts().reindex(range(12), fill_value=0)
        
        # Create a DataFrame for easier sorting and mapping
        key_df = pd.DataFrame({
            'Key': range(12),
            'Count': key_counts.values,
            'Alphabetical Key': key_mapping
        })

        # Sort the DataFrame by key count in descending order
        key_df = key_df.sort_values(by='Count', ascending=False)

        # Create a bar chart with the key counts
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        sns.barplot(x='Alphabetical Key', y='Count', data=key_df, palette='plasma', ax=ax)
        # Set y-axis ticks to integers
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_facecolor('lightgrey')
        plt.xlabel('Key')
        plt.ylabel('Count')
        fig.patch.set_facecolor('lightgrey')

        return fig
    
    def mode_pie_chart(self) -> plt.Figure:
        # Count values for each mode category
        major_count = self.df_album['mode'].sum()  # mode 1 is major
        minor_count = len(self.df_album) - major_count

        mode_counts = [major_count, minor_count]
        labels = ['Major', 'Minor']
        # Colors
        color_major = cm.plasma(0.10)
        color_minor = cm.plasma(0.65)
        colors_with_alpha = [(color_major[0], color_major[1], color_major[2], 0.7),  # 70% opacity for Major
                            (color_minor[0], color_minor[1], color_minor[2], 0.7)]  # 70% opacity for Minor

        fig, ax = plt.subplots(figsize=(6, 4))
        patches, texts, autotexts = ax.pie(mode_counts,
                                        labels=labels,
                                        colors=colors_with_alpha,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        textprops={'fontsize': 12},
                                        radius=1.2)
        
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()

        return fig
    
    def explicit_pie_chart(self) -> plt.Figure:
        # Count values for each category
        explicit_counts = self.df_album['is_explicit'].sum()
        non_explicit_count = len(self.df_album) - explicit_counts

        explicit_counts = [explicit_counts,non_explicit_count]
        labels = ['Explicit','Non-Explicit']
        # Colors
        color_explicit = cm.plasma(0.10)
        color_non_explicit = cm.plasma(0.65)
        colors_with_alpha = [(color_explicit[0], color_explicit[1], color_explicit[2], 0.7),  # 70% opacity for Explicit
                             (color_non_explicit[0], color_non_explicit[1], color_non_explicit[2], 0.7)]  # 70% opacity for Non-Explicit
        
        fig, ax = plt.subplots(figsize=(6, 4))
        patches, _,_= ax.pie(explicit_counts,
                             labels=labels,
                             colors=colors_with_alpha,
                             autopct='%1.1f%%',
                             startangle=90,
                             textprops={'fontsize': 12},
                             # shadow=True,
                             radius=1.2)
        ax.axis('equal')

        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')

        return fig
    
    def album_popularity_gauge_chart(self) -> go.Figure:
        """ 
         Create a gauge chart displaying the mean popularity score of an album
         
         Returns:
         go.Figure: Gauge chart
           """
        def calculate_popularity():
            return self.df_album['popularity'].mean()
        
        mean_popularity = calculate_popularity()

        # Create a gauge chart
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=mean_popularity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "The Current Popularity"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth':1, 'tickcolor': "darkblue"},
                'bar': {'color': 'rgba(89, 42, 154, 1)'},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                {'range': [0, 100], 'color': 'lightgrey'},
                {'range': [0, mean_popularity], 'color': 'rgba(65, 105, 225, 0.5)'}
                ],
            }
        ))
        fig.update_layout(
            paper_bgcolor='lightgrey',
            font={'color':"black"},
            height=450,
            width=700
        )

        return fig

    
    def run_analysis(self) -> None:
        try:
            st.header('Album DataFrame')
            st.dataframe(self.df_album)

            st.header('Attribute Radar Chart')
            st.text("The radar chart displays various musical attributes of the album to compare their relative strengths.")
            fig = self.radar_chart()
            st.plotly_chart(fig)

            st.header('Tempo Histogram')
            st.text("The tempo histogram shows the distribution of the tempo (beats per minute) across all tracks in the album.")
            fig = self.tempo_histogram()
            st.pyplot(fig)

            st.header('Duration Histogram')
            st.text("The histogram illustrates the distribution of track durations within the album, highlighting variability in song lengths.")
            fig = self.duration_histogram()
            st.pyplot(fig)

            st.header('Loudness Histogram')
            st.text("The loudness histogram plots the loudness levels (in decibels) of each track, showing the dynamic range of the album.")
            fig = self.loudness_histogram()
            st.pyplot(fig)

            st.header('Key Histogram')
            st.text("The histogram displays the musical keys of the album's tracks, indicating the most common keys used.")
            fig = self.key_histogram()
            st.pyplot(fig)

            st.header("Mode Pie Chart")
            st.text("Major modes are bright and uplifting, while minor modes are somber and serious.")
            mode_pie = self.mode_pie_chart()
            st.pyplot(mode_pie)

            st.header('Explicitness Pie Chart')
            st.text("The pie chart breaks down the proportion of explicit to non-explicit tracks, providing insight into the album's content.")
            fig = self.explicit_pie_chart()
            st.pyplot(fig)

            st.header('Album Popularity Gauge Chart')
            st.text("The gauge chart displays the album's popularity score, "
                    "which is the average of the popularity scores of all tracks in the album.")
            fig = self.album_popularity_gauge_chart()
            st.plotly_chart(fig)



        
        except Exception as e:
            print(f" Run Analysis Error {e}")
            st.error("An error occurred during analysis. Please try again.")


# Initialize Spotify Client
def init_spotify_client():
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]  #config('SPOTIFY_CLIENT_ID')
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"] # config('SPOTIFY_CLIENT_SECRET')
    credential_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=credential_manager)
    return sp
sp = init_spotify_client()


# Album Search Function
def artist_search_func(sp,query) -> List[str]:
    result = sp.search(q=query, type='artist', limit=5)
    artists = [artist['name'] for artist in result['artists']['items']]
    return artists

def album_search_func(sp, artist_name, query) -> List[Tuple[str, str]]:
    # Find the artist's Spotify ID based on their name.
    artist_result = sp.search(q=f"artist:{artist_name}", type='artist', limit=1)
    if not artist_result['artists']['items']:
        return []  # Return an empty list if the artist is not found.

    artist_id = artist_result['artists']['items'][0]['id']
    
    # Use the artist's Spotify ID to search for albums.
    albums = []
    album_result = sp.artist_albums(artist_id, album_type='album', limit=20)
    for album in album_result['items']:
        # Check if the album name contains the query string (case insensitive).
        if query.lower() in album['name'].lower():
            albums.append((album['name'], album['id']))
    
    return albums


with st.sidebar:
    st.title("Select Album Name")
    selected_artist_name = st.text_input("Artist Name")

    album_options = album_search_func(sp, selected_artist_name, "")
    album_name_to_id = {name: id for name, id in album_options}

    # Placeholder option
    placeholder = "Select an album..."
    # Prepare selectbox options with the placeholder
    options = [placeholder] + list(album_name_to_id.keys())

    selected_album_name = st.selectbox("Select Album", options=options, index=0)

    # Check if a valid album is selected
    if selected_album_name != placeholder:
        selected_album_id = album_name_to_id[selected_album_name]
        print(f"Selected Album ID: {selected_album_id}")
    else:
        selected_album_id = None

analyze_button = st.sidebar.button("Analyze")

#-------- Main ----------
st.markdown("# Album Analysis")
st.info("Select an artist and album name. You'll get a comprehensive analysis of your selected album.", icon="📀")

try:
    if analyze_button:
        album_analyzer = AlbumAnalyzer(sp, selected_album_id)
        print(album_analyzer)
        st.components.v1.iframe(f"https://open.spotify.com/embed/album/{selected_album_id}?utm_source=generator&theme=0",
                        width=500, height=160, scrolling=True)
        st.divider()

        # Run Analysis
        album_analyzer.run_analysis()

except Exception as e:
    print(e)
    st.error(f'An error occurred: {str(e)}')

            

