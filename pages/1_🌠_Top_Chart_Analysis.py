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
st.set_page_config(page_title="Top Chart Analysis", 
                   page_icon="💿")

# Hot Chart playlist options
playlists = {
    "Select a Playlist": None, # Placeholder value
    "Billboard Hot 100": "6UeSakyzhiEt4NB3UAd6NQ",
    "Top 50 Global (Daily)": "37i9dQZEVXbMDoHDwVN2tF",
    "Top Songs Global (Weekly)": "37i9dQZEVXbNG2KDcFcKOF",
    "Big On Ineternet": "37i9dQZF1DX5Vy6DFOcx00",
    "Viral 50 Global (Daily)": "37i9dQZEVXbLiRSasKsNU9"
    
}

# Playlist ttl mapping
playlists_ttl = {
    "Billboard Hot 100": ("6UeSakyzhiEt4NB3UAd6NQ", 604800), # 1 week TTL
    "Top 50 Global (Daily)": ("37i9dQZEVXbMDoHDwVN2tF", 86400), # 1 day TTL
    "Top Songs Global (Weekly)": ("37i9dQZEVXbNG2KDcFcKOF", 604800), # 1 week TTL
    "Big On Ineternet": ("37i9dQZF1DX5Vy6DFOcx00", 86400), # 1 day TTL
    "Viral 50 Global (Daily)": ("37i9dQZEVXbLiRSasKsNU9", 86400) # 1 day TTL
}

# Retrieve playlist data
def retrieve_latest_data(_sp, playlist_id: str)-> Tuple[pd.Series, pd.DataFrame]:
    """ 
     Retrieves and processes the latest tracks data from a Spotify playlist
     Playlist id
     Top Global 50 Songs (Weekly Update)
     Args:
        _sp (spotipy.Spotify): Spotify API client
        playlist_id (str, optional): Spotify playlist id. Defaults to '37i9dQZEVXbNG2KDcFcKOF'.
     Returns:
        A tuple containing two elements:
        - A pandas Series with the mean values of selected audio features
        - A pandas DataFrame containing detailed info about each track in the playlist

       """
    try:
        # Search TTL in the mapping
        ttl = None
        for _, (id, t) in playlists_ttl.items():
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

class SpotifyAnalyzer:
    def __init__(self, sp, playlist_id: str) -> None:
        self.sp = sp
        self.mean_values_top_50, self.df_top_50 = retrieve_latest_data(self.sp, playlist_id)

    def radar_chart(self) -> go.Figure:
        color_top_50 = 'rgba(93, 58, 155, 0.9)' 
        means_values_top_50p = self.mean_values_top_50 * 100
        att_list = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
        fig = go.Figure()

        # Add trace (mean)
        fig.add_trace(go.Scatterpolar(
            r=means_values_top_50p,
            theta=att_list,
            fill='toself',
            name='Playlist (mean)',
            fillcolor=color_top_50,
            line=dict(color=color_top_50),
        ))

        # Update the layout 
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            legend=dict(x=1, y=1, traceorder='normal',orientation='v', font=dict(color='black')),
            paper_bgcolor = 'lightgrey',
            font = {"color": "black"},
            height=450,
            width=700
        )

        return fig
        
    def tempo_histogram(self) -> plt.Figure:
        color_top_50 = cm.plasma(0.15)
        color_average_tempo = cm.plasma(0.55) 

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df_top_50['tempo'], 
                     bins=50, 
                     kde=True, 
                     color=color_top_50, 
                     edgecolor='black', 
                     ax=ax)
        mean_tempo = self.df_top_50['tempo'].mean()
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

        ax.grid(False, axis='x')
        ax.grid(True, axis='y', linestyle='--',alpha=0.6)

        # Set background color
        fig.patch.set_facecolor('lightgrey')
        ax.set_facecolor('lightgrey')

        return fig
    
    def duration_histogram(self) -> plt.Figure:
        color_top_50 = cm.plasma(0.15)
        color_average_duration = cm.plasma(0.55) 

        # Convert duration from milliseconds to minutes for readability
        self.df_top_50['duration_min'] = self.df_top_50['duration_ms'] / 60000

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df_top_50['duration_min'], 
                    bins=50, 
                    kde=True, 
                    color=color_top_50, 
                    edgecolor='black', 
                    ax=ax)
        mean_duration = self.df_top_50['duration_min'].mean()
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

        # Set background color
        fig.patch.set_facecolor('lightgrey')
        ax.set_facecolor('lightgrey')

        return fig

    
    def loudness_histogram(self) -> plt.Figure:
        color_top_50 = cm.plasma(0.15)
        color_average_loud = cm.plasma(0.55) 

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df_top_50['loudness'], 
                     bins=50, 
                     kde=True, 
                     color=color_top_50, 
                     edgecolor='black', 
                     ax=ax)
        mean_tempo = self.df_top_50['loudness'].mean()
        ax.axvline(mean_tempo, 
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
        ax.grid(True, axis='y', linestyle='--',alpha=0.6)

        # Set background color
        fig.patch.set_facecolor('lightgrey')
        ax.set_facecolor('lightgrey')

        return fig
    
    def key_distribution_chart(self) -> plt.Figure:
        # Mapping of numeric key values to corresponding alphabetic keys
        key_mapping = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

        # Get key counts and ensure all keys are present
        key_counts = self.df_top_50['key'].value_counts().reindex(range(12), fill_value=0)
        
        # Create a DataFrame for easier sorting and mapping
        key_df = pd.DataFrame({
            'Key': range(12),
            'Count': key_counts.values
        })

        # Sort the DataFrame by 'Count'
        key_df_sorted = key_df.sort_values(by='Count', ascending=False)

        # Map numeric keys to their names
        key_df_sorted['Key Name'] = key_df_sorted['Key'].apply(lambda x: key_mapping[x])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # Plot using the sorted DataFrame
        sns.barplot(x='Key Name', y='Count', data=key_df_sorted, palette='plasma', ax=ax)

        ax.set_facecolor('lightgrey')
        plt.xlabel('Key')
        plt.ylabel('Count')
        fig.patch.set_facecolor('lightgrey')

        return fig
    
    def explicit_pie_chart(self) -> plt.Figure:
        # Count values for each category
        explicit_counts = self.df_top_50['is_explicit'].sum()
        non_explicit_count = len(self.df_top_50) - explicit_counts

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
        plt.tight_layout()

        return fig

    def genres_wordcloud(self) -> plt.Figure:
        def clean_genres(genres_str):
            genres_list = genres_str.split(',')
            # print(f"genres list {genres_list}")
            # cleand_genres = [genre.strip() for genre in genres_list if genre.strip() != '']
            cleaned_genres = []
            for genre in genres_list:
                stripped = genre.strip()
                if stripped != '':
                    cleaned_genres.append(stripped)
            cleaned_genres = ', '.join(cleaned_genres)
            # print(f"cleaned genres list {cleaned_genres}")
            return cleaned_genres
        # Check the dataframe
        if self.df_top_50.empty:
            # print("DataFrame is empty. Cannot generate genres word cloud.")
            return
        all_genres = ', '.join(self.df_top_50['genres'])
        cleaned_all_genres = clean_genres(all_genres)

        # Generate word cloud
        regexp = r"\w(?:[-']?\w)+"
        wc = WordCloud(width=700, 
                              height=500,
                              background_color='white',
                              stopwords=None,
                              colormap='plasma_r',
                              collocations=True,
                              scale=2,
                              regexp=regexp,
                              max_words=200,
                              min_font_size=8).generate(cleaned_all_genres)
        # print(wc)
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(7,5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout(pad=0)

        return fig
    
        
    # Run Analysis & Render Visualizations
    def run_analysis(self) -> None:
        try:
            # Create a DataFrame
            st.header('Playlist DataFrame')
            st.dataframe(self.df_top_50)

            # Create a Radar Chart
            st.header('Attributes Radar Chart')
            st.text("The radar chart displays the distribution of various musical attributes for the selected tracks.")
            fig = self.radar_chart()
            st.plotly_chart(fig)

            # Create a BPM Histogram Chart
            st.header('Tempo Histogram Chart')
            st.text("This histogram shows the distribution of tempo (beats per minute) across tracks.")
            bpm_hist_chart = self.tempo_histogram()
            st.pyplot(bpm_hist_chart)

            st.header('Duration Histogram Chart')
            st.text("The histogram below represents the distribution of track durations in the playlist.")
            duration_dist = self.duration_histogram()
            st.pyplot(duration_dist)

            # Create a Key Distribution
            st.header('Key Distribution Comparison:')
            st.text("This chart compares the key distribution of the tracks, showing which musical keys are most common.")
            key_dist = self.key_distribution_chart()
            st.pyplot(key_dist)

            # Create a Duration Histogram Chart
            st.header('Loudness Histogram Chart')
            st.text("The loudness histogram visualizes the loudness levels of tracks in decibels.")
            loudness_hist_chart = self.loudness_histogram()
            st.pyplot(loudness_hist_chart)

            # Create an Explicit Pie Chart
            st.header('Explicitness Pie Chart')
            st.text("This pie chart shows the proportion of explicit and non-explicit tracks in the playlist.")
            explicit_chart = self.explicit_pie_chart()
            st.pyplot(explicit_chart)

            # Create a Genre Word Cloud
            st.header('Genres Word Cloud')
            st.text("The word cloud illustrates the prevalence of various genres in the playlist based on text data.")
            wc = self.genres_wordcloud()
            st.pyplot(wc)


        except Exception as e:
            st.error("An error occurred during analysis. Please try again.")


# Initialize Spotify Client
def init_spotify_client():
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]  #config('SPOTIFY_CLIENT_ID')
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"] # config('SPOTIFY_CLIENT_SECRET')
    credential_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=credential_manager)
    return sp
sp = init_spotify_client()
   


# Sidebar
with st.sidebar:
    st.header("Top Chart Analysis")
    # Playlist selection
    selected_playlist = st.selectbox("Select a Playlist", options=list(playlists.keys()))
    # Obtain playlist id
    playlist_id = playlists[selected_playlist]
    # Compare button
    compare_button = st.sidebar.button("Compare")
    

##---- Main ------
st.markdown("# Top Chart Analysis")
# Header Text
st.info("Get the Latest Top Chart on Spotify, updated daily/weekly, and experience the data of top-performing tracks.", icon="👩🏽‍🎤")

if compare_button:
    try:
        spotify_analyzer = SpotifyAnalyzer(sp, playlist_id)
        st.components.v1.iframe(f"https://open.spotify.com/embed/playlist/{playlist_id}?utm_source=generator&theme=0",
                        width=500, height=160, scrolling=True)
        st.divider()

        # Run Analysis
        spotify_analyzer.run_analysis()
        st.balloons()
    
    except Exception as e:
        print(e)
        st.error(f'An error occurred: {str(e)}')

st.divider()
st.caption("🛑Disclaimer: This application is designed for educational and entertainment purposes only. The data used in this app is sourced from public APIs and may not be entirely accurate or up-to-date. The analysis and visualizations provided should not be considered professional advice, and the creators of this app are not responsible for any decisions made based on the information provided. Please note that this app is not affiliated with Spotify or any other third-party services. Use this app responsibly and be aware of the terms and conditions of the Spotify API. Enjoy exploring the data responsibly!")

