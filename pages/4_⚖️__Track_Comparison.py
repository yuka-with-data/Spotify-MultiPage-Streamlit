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
st.set_page_config(page_title="Track-To-Chart Comparison", 
                   page_icon="⚖️")

# Playlist ttl mapping
playlists_ttl = {
    "Billboard Hot 100": ("6UeSakyzhiEt4NB3UAd6NQ", 604800), # 1 week TTL
    "Top 50 Global (Daily)": ("37i9dQZEVXbMDoHDwVN2tF", 86400), # 1 day TTL
    "Top Songs Global (Weekly)": ("37i9dQZEVXbNG2KDcFcKOF", 604800), # 1 week TTL
    "Big On Ineternet": ("37i9dQZF1DX5Vy6DFOcx00", 86400), # 1 day TTL
    "Viral 50 Global (Daily)": ("37i9dQZEVXbLiRSasKsNU9", 86400) # 1 day TTL
}

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

# Class
class SpotifyAnalyzer:
    def __init__(self, sp, playlist_id:str) -> None:
        self.sp = sp
        # Load Playlist Data
        self.mean_values_top_50, self.df_top_50 = retrieve_latest_data(self.sp, playlist_id)

    #@st.cache_data
    def get_spotify_data(_self, artist:str, track:str) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        """ 
         Obtain Spotify data for a given artist and track by the user
         Args:
           artist (str): The name of the artist
           track (str): The name of the track
         Returns:
           track attributes
           access token
           """
        # Search for the track using the Spotify API
        result = _self.sp.search(q=f'artist:{artist} track:{track}', type='track', limit=1)
        #result = self.sp.search(q=f'{artist} {track} -{artist} -cover -remix -parody -piano -Live', type='track', limit=1)
        print(f"Search result: {result}")

        # Check for successful response for the track search
        if not result or not result['tracks']['items']:
            # print(f"No exact match found for '{track}' by '{artist}'")
            st.error("Error: The right track cannot be found under the chosen artist. Please check your input.")
            return None, None

        # Save a found track
        found_track = result['tracks']['items'][0]

        # Check if the found track's name is the same as the expected track name
        if track.lower() != found_track['name'].lower():
            st.error(f"Close match found. Did you mean '{found_track['name']}' instead of '{track}'")
            return None, None

        # Retrieve audio features 
        audio_features_response = _self.sp.audio_features(found_track['id'])

        # Check if the response is valid and contains the expected dictionary
        if not audio_features_response or not isinstance(audio_features_response, list):
            print("Error fetching audio features: Invalid response format")
            return None, None

        audio_features_data = audio_features_response[0]

        # Check if the response is a valid dictionary
        if not isinstance(audio_features_data, dict):
            print("Error fetching audio features: Invalid response format")
            return None, None
        
        # Genres
        artist_id = found_track['artists'][0]['id']
        artist_info = _self.sp.artist(artist_id)
        genres = artist_info.get('genres', '')

        # Extract the 'is_explicit' attribute from the found track
        is_explicit = found_track.get('explicit', False)
        popularity = found_track.get('popularity', None)
        track_id = found_track['id']

        # Extract audio features data from the response
        list_att = ["danceability", 
                    "valence", 
                    "energy", 
                    "acousticness", 
                    "instrumentalness", 
                    "liveness", 
                    "speechiness", 
                    "key", 
                    "tempo", 
                    "duration_ms", 
                    "loudness"]
        extracted_attributes = {
            key: audio_features_data.get(key, None) for key in list_att
        }
        extracted_attributes.update({
                "popularity": popularity,
                "id": track_id,
                'is_explicit': is_explicit,
                'genres': ', '.join(genres) # Convert list of genres to a comma-seperated string
                })

        # Optional: Save the audio features data to JSON file or do further processing
        with open('audio_features.json', 'w', encoding='utf-8') as json_file:
            json.dump(audio_features_data, json_file, ensure_ascii=False, indent=4)

        # Return 2 objects: extracted_attributes and access_token
        return extracted_attributes


    # Create the rader chart
    def create_radar_chart(self, audio_features: Dict[str, float], artist_name: str, track_name: str) -> go.Figure:
        """ 
         Create a radar chart comparing audio features of a track with the top 100
         Args:
           audio_features (dict)
           artist_name (str)
           track_name (str)
         Returns:
           radar chart
           """

        color_top_50 = 'rgba(93, 58, 155, 0.9)' 
        color_your_track = 'rgba(230, 97, 0, 0.7)'

        means_values_top_50p = self.mean_values_top_50 * 100
        audio_feat_percent = {key: value * 100 for key, value in audio_features.items()}
        audio_feat_values = list(audio_feat_percent.values()) # Convert to a list
        att_list = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
        fig = go.Figure()

        # Add trace for the Top 100 songs (mean)
        fig.add_trace(go.Scatterpolar(
            r=means_values_top_50p,
            theta=att_list,
            fill='toself',
            name='Playlist (mean)',
            fillcolor=color_top_50,
            line=dict(color=color_top_50),
        ))
        # Add trace for the Your Favorite Song
        fig.add_trace(go.Scatterpolar(
            r=audio_feat_values,
            theta=att_list,
            fill='toself',
            name=f'Your Track',
            fillcolor=color_your_track,
            line=dict(color=color_your_track, dash='dot'),
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
    
    def create_bpm_histogram(self, audio_features: Dict[str, float]) -> plt.Figure:
        """ 
         Create a histogram chart comparing BPM(tempo) of a track with the top 100
         Args:
           audio_features (dict)
         Returns:
           histogram chart
           """
        # Use Plasma colormap colors
        color_top_50 = cm.plasma(0.15)
        color_your_track = cm.plasma(0.65)
        color_average_tempo = cm.plasma(0.55) 

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df_top_50['tempo'], bins=50, kde=True, color=color_top_50, edgecolor='black', label=f"{selected_playlist}", ax=ax)
        ax.axvline(audio_features['tempo'], color=color_your_track, linewidth=2, label='Your Track')
        
        mean_tempo = self.df_top_50['tempo'].mean()
        ax.axvline(mean_tempo, color=color_average_tempo, linestyle='dashed', linewidth=2, label='Average Tempo')
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Frequency')
        ax.legend(prop={'size': 8})
        
        # Set y-axis ticks to integers
        # Determine max y-value (rounded up) and set y-ticks accordingly
        max_count = int(max(ax.get_yticks())) # Find the current max y-tick and round up
        ax.set_yticks(range(0, max_count))

        ax.grid(False, axis='x')
        ax.grid(True, axis='y', linestyle='--',alpha=0.6)

        # Set background color
        fig.patch.set_facecolor('lightgrey')
        ax.set_facecolor('lightgrey')

        return plt

    def create_key_distribution_chart(self, audio_features:Dict[str,float], track_name: str) -> plt.Figure:
        """ 
            Retrive the key for selected artist and track.
            Args:
            artist_name (str)
            track_name (str)
            Returns:
            key (str) or None
            """
        # Mapping of numeric key values to corresponding alphabetic keys
        key_mapping = ('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')

        # Get the key distribution for the top 50 tracks
        # Sort the key counts in descending order by count values
        key_counts = self.df_top_50['key'].value_counts().sort_values(ascending=False).reindex(range(12), fill_value=0)
        print(f"Key Counts Ordered: {key_counts}")

        # re-sort the key counts series after the 'reindex' operation
        key_counts = key_counts.sort_values(ascending=False)
        print(f"Key Counts Re-Ordered: {key_counts}")

        # Initialize figure and axes objects
        fig, ax = plt.subplots(figsize=(6,4))

        # Use seaborn to set the plotting style
        sns.set_theme(style="whitegrid")

        # FutureWarning: sns
        if isinstance(key_counts.index.dtype, pd.CategoricalDtype):
            ax = sns.barplot(x=key_counts.index.codes,# Use codes for categorical data
                             y=key_counts.values,
                             palette='plasma',
                             order=key_counts.index)
        else:
            ax = sns.barplot(x=key_counts.index,
                             y=key_counts.values,
                             palette='plasma',
                             order=key_counts.index)

        # Map numeric key values to corresponding alphabetic keys for x-axis tick lables
        alpha_keys = [key_mapping[num] for num in key_counts.index if num != -1]
        ax.set_xticklabels(alpha_keys)

        # Adjust y-axis to increment by 1
        max_key_count = key_counts.max()
        ax.set_yticklabels(np.arange(0,max_key_count + 1, 1)) # +1 to ensure the max value is included

        # Numeric Key from Target Track
        key_index = audio_features['key']
        if 0 <= key_index <= len(key_mapping):
            selected_key = key_mapping[key_index]
            print(f'Selected Key from User Input: {selected_key}')
            # Get reordered index of the selected key in the dataset
            selected_key_index = key_counts.index.get_loc(key_index)
            print(f'Selected Key Index in Reordered Distribution: {selected_key_index}')

            for i, patch in enumerate(ax.patches):
                if i != selected_key_index:
                    # fade the rest of the bars
                    patch.set_alpha(0.2)

            st.write(f"<p style='font-size:24px'>Key for '{track_name}': {selected_key}</p>", unsafe_allow_html=True)
        else:
            print(f"Key '{key_index}' not found in key distribution")

        ax.set_facecolor('lightgrey')

        # plt.title('Distribution of Songs by Key: 2023 Top 50')
        plt.xlabel('Key')
        plt.ylabel('Count')

        fig.patch.set_facecolor('lightgrey')
        
        return plt
    
    def create_duration_histogram(self, audio_features: Dict[str, float]) -> plt.Figure:
        """ 
        Create a histogram chart comparing BPM(tempo) of a track with the top 100
        Args:
        audio_features (dict)
        Returns:
        histogram chart
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        # Convert duration from miliseconds to seconds
        duration_sec = self.df_top_50['duration_ms'] / 1000
        audio_duration_sec = audio_features['duration_ms'] / 1000
        mean_duration_sec = self.df_top_50['duration_ms'].mean() / 1000

        color_top_50 = cm.plasma(0.15)
        color_your_track = cm.plasma(0.7)
        color_average_duration = cm.plasma(0.55)

        sns.histplot(duration_sec, bins=50, kde=True, alpha = 0.7, color=color_top_50, edgecolor='black', label=f"{selected_playlist}", ax=ax)
        ax.axvline(audio_duration_sec, color=color_your_track, linewidth=2, label='Your Track')
        ax.axvline(mean_duration_sec, color=color_average_duration, linestyle='dashed', linewidth=2, label='Average Track Duration')

        ax.set_xlabel('Duration (in seconds)')
        ax.set_ylabel('Frequency')
        ax.legend(prop={'size': 8})
        ax.grid(False, axis='x')
        ax.grid(True, axis='y', linestyle='--',alpha=0.6)

        # Ensure y-axis ticks are integers
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        fig.patch.set_facecolor('lightgrey')
        ax.set_facecolor('lightgrey')

        return plt

    def create_loudness_histogram(self, audio_features: Dict[str, float]) -> plt.Figure:
        """ 
        Create a histogram chart comparing loudness of a track with the top 50
        Args:
        audio_features (dict)
        Returns:
        histogram chart
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        color_top_50 = cm.plasma(0.15)
        color_your_track = cm.plasma(0.7)
        color_average_duration = cm.plasma(0.55)

        sns.histplot(self.df_top_50['loudness'], bins=50, kde=True, alpha = 0.7, color=color_top_50,edgecolor='black', label=f"{selected_playlist}", ax=ax)
        ax.axvline(audio_features['loudness'], color=color_your_track, linewidth=2, label='Your Track')
        ax.axvline(self.df_top_50['loudness'].mean(), color=color_average_duration, linestyle='dashed', linewidth=2, label='Average Track Loudness')

        ax.set_xlabel('Loudness (in dB)')
        ax.set_ylabel('Frequency')

        # Determine max y-value (rounded up) and set y-ticks accordingly
        max_count = int(max(ax.get_yticks()))  # Find the current max y-tick and round up
        ax.set_yticks(range(0, max_count))

        ax.legend(prop={'size': 8})
        ax.grid(False, axis='x')
        ax.grid(True, axis='y', linestyle='--',alpha=0.6)
        
        fig.patch.set_facecolor('lightgrey')
        ax.set_facecolor('lightgrey')

        return plt


    def create_explicit_pie_chart(self, audio_features: Dict[str, float]) -> plt.Figure:
        """
        Create a pie chart displaying the percentage of explicit tracks in the top 50
        Returns:
        pie chart
        """
        # Extract the attribute from user's selected track 
        is_explicit = audio_features.get('is_explicit', False)

        # Count values for each category
        explicit_counts = self.df_top_50['is_explicit'].sum()
        non_explicit_count = len(self.df_top_50) - explicit_counts

        explicit_counts = [explicit_counts,non_explicit_count]
        labels = ['Explicit','Non-Explicit']
        colors = [cm.plasma(0.10), cm.plasma(0.65)]
        explode = [0.02, 0] if is_explicit else [0,0.02]

        fig, ax = plt.subplots(figsize=(6, 4))
        patches, _,_= ax.pie(explicit_counts, 
                            labels=labels,
                            colors=colors, 
                            autopct='%1.1f%%', 
                            startangle=90, 
                            explode=explode, 
                            textprops={'fontsize': 12}, 
                            # shadow=True, 
                            radius=1.2)
        
        if is_explicit:
            patches[0].set_alpha(0.65)
            patches[1].set_alpha(0.5)
        else:
            patches[0].set_alpha(0.25)
            patches[1].set_alpha(1)

        # Convert the boolean to a readable string
        explicit_status = "Explicit" if is_explicit else "Non-Explicit"
        st.write(f"<p style='font-size:24px'>Explicitness Status for '{selected_track}': {explicit_status}</p>", unsafe_allow_html=True)
        ax.axis('equal')
        plt.tight_layout()

        return fig
    
    def create_mode_pie_chart(self, audio_features: Dict[str, float]) -> plt.Figure:
        """
        Create a pie chart displaying the percentage of tracks in Major and Minor modes in the top chart
        and highlight the mode of the selected track.
        Returns:
        pie chart
        """
        # Extract the mode attribute from the user's selected track
        is_major = audio_features.get('mode', 1) == 1  # '1' for Major and '0' for Minor

        # Count values for each mode category in the playlist
        major_count = self.df_top_50['mode'].sum()
        minor_count = len(self.df_top_50) - major_count

        mode_counts = [major_count, minor_count]
        labels = ['Major', 'Minor']
        colors = [cm.plasma(0.10), cm.plasma(0.65)]
        explode = [0.02, 0] if is_major else [0, 0.02]  # Highlight the selected track's mode

        fig, ax = plt.subplots(figsize=(6, 4))
        patches, _, _ = ax.pie(mode_counts,
                            labels=labels,
                            colors=colors,
                            autopct='%1.1f%%',
                            startangle=90,
                            explode=explode,
                            textprops={'fontsize': 12},
                            radius=1.2)
        
        if is_major:
            patches[0].set_alpha(0.65)
            patches[1].set_alpha(0.5)
        else:
            patches[0].set_alpha(0.25)
            patches[1].set_alpha(1)

        # Convert the boolean to a readable string
        mode_status = "Major" if is_major else "Minor"
        st.write(f"<p style='font-size:24px'>Mode Status for '{selected_track}': {mode_status}</p>", unsafe_allow_html=True)
        ax.axis('equal')
        plt.tight_layout()

        return fig



    def create_genres_wordcloud(self) -> plt.Figure:
        """
        Create a Word Cloud displaying the genre overview for the Top Chart playlist
        Returns:
        word cloud
        """
        # Inner function to clean the genres str
        def clean_genres(genres_str):
            genres_list = genres_str.split(',')
            print(f"genres list {genres_list}")
            # cleand_genres = [genre.strip() for genre in genres_list if genre.strip() != '']
            cleaned_genres = []
            for genre in genres_list:
                stripped = genre.strip()
                if stripped != '':
                    cleaned_genres.append(stripped)
            cleaned_genres = ', '.join(cleaned_genres)
            print(f"cleaned genres list {cleaned_genres}")
            return cleaned_genres
        # Check the dataframe
        if self.df_top_50.empty:
            print("DataFrame is empty. Cannot generate genres word cloud.")
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
        print(wc)
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(7,5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout(pad=0)

        print(fig)
        return fig
                            

    def create_popularity_chart(self, audio_features: Dict[str, float]) -> go.Figure:
        """ 
         Create a popularity gauge chart displaying the current popularity score of a track
         Args:
           audio_features (dict)
         Returns:
           gauge chart
           self.df_top_50
           """
        # Loading popularity data from Top 50
        # _, df = self.load_stream_data()
        popularity_mean = self.df_top_50['popularity'].mean()

        # Get the popularity score for the target track
        popularity = audio_features.get('popularity', None)
        if popularity is not None:
            # Create a gauge chart
            fig = go.Figure()

            fig.add_trace(go.Indicator(
                mode = "gauge+number",
                value = popularity,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Popularity Score"},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': 'rgba(89, 42, 154, 1)'},
                    'bgcolor': "black",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 100], 'color': 'white'},
                        {'range': [0, popularity], 'color': 'rgba(65, 105, 225, 0.5)'}
                    ],
                    'threshold': {
                        'line': {'color': 'rgba(255, 140, 0, 1.0)', 'width': 4},
                        'thickness': 0.8,
                        'value': popularity_mean
                    }
                }
            ))

            # Add annotation for the threashold line
            fig.add_annotation(x=0.78, y=0.13, xref="paper", yref="paper",
                           text=f"Playlist Average: {popularity_mean:.2f}",
                           showarrow=False, font=dict(size=14, color="rgba(195, 107, 0, 1.0)"))

            fig.update_layout(
                paper_bgcolor="lightgrey",
                font={"color": "black"},
                height=450,
                width=700
            )

        else:
            st.warning("Popularity data not available for this track.")

        return fig

    
    def run_analysis(self, artist_name: str, track_name: str) -> None:
        """ 
         Run a comprehensive analysis for a given artist and track name
         Args:
           artist_name (str)
           track_name (str)
         Returns:
           None
           """
        try:
            # Get Spotify Data
            audio_features= self.get_spotify_data(artist_name, track_name)
            print(f'Audio Features: {audio_features}')
            track_id = audio_features.get('id', None)

            # User's Track Player
            st.header('Your Track')
            st.components.v1.iframe(f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator",
                                                width=500, height=160, scrolling=True)

            if audio_features:
                # Create a Radar Chart
                st.header('Radar Chart Comparison:')
                st.text(f'Comparison of Attributes in % {selected_playlist} vs. {track_name} by {artist_name}')
                fig = self.create_radar_chart(audio_features, artist_name, track_name)
                st.plotly_chart(fig)

                # Create a BPM Histogram Chart
                st.header('Histogram Chart Comparison - Tempo:')
                st.text(f'Comparison of Tempo in % {selected_playlist} vs. {track_name} by {artist_name}')
                bmp_hist_chart = self.create_bpm_histogram(audio_features)
                st.pyplot(bmp_hist_chart)

                # Show Key Distribution
                st.header('Key Distribution Comparison:')
                st.text(f'Distribution of Songs by Key: {selected_playlist} vs. {track_name} by {artist_name}')
                key_dist = self.create_key_distribution_chart(audio_features, track_name)
                st.pyplot(key_dist)

                # Create a Duration Histogram Chart
                st.header('Histogram Chart Comparison - Duration:')
                st.text(f'Comparison of Duration {selected_playlist} vs. {track_name} by {artist_name}')
                duration_chart = self.create_duration_histogram(audio_features)
                st.pyplot(duration_chart)

                # Create a Loudness Histogram Chart
                st.header('Histogram Chart Comparison - Loudness:')
                st.text(f'Comparison of Loudness {selected_playlist} vs. {track_name} by {artist_name}')
                loudness_chart = self.create_loudness_histogram(audio_features)
                st.pyplot(loudness_chart)

                # Create a Mode Pie Chart
                st.header('Mode Pie Chart:')
                st.text(f'Comparison of Mode for {selected_playlist} & {track_name} by {artist_name}')
                mode_chart = self.create_mode_pie_chart(audio_features)
                st.pyplot(mode_chart)

                # Create a Explicit Pie Chart
                st.header('Explicitness Pie Chart:')
                st.text(f'Comparison of Explicitness {selected_playlist} & {track_name} by {artist_name}')
                explicit_chart = self.create_explicit_pie_chart(audio_features)
                st.pyplot(explicit_chart)

                # Create a Genres Word Cloud
                st.header("Genres Word Cloud:")
                st.text(f'Genres Word Cloud for {selected_playlist}')
                user_genres = audio_features.get('genres', '')
                if user_genres:
                    st.write(f"<p style='font-size:24px'>{selected_artist}'s Genres: {user_genres}</p>", unsafe_allow_html=True)
                else:
                    st.error("No genres data available for the selected track/artist")
                fig = self.create_genres_wordcloud()
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.error("Word Cloud could not be generated due to the insufficient data.")
                

                # Create a Popularity Gauge Chart
                st.header('Popularity Gauge Chart:')
                st.text(f'Current Popularity Score for {track_name} by {artist_name}')
                pop_chart = self.create_popularity_chart(audio_features)
                st.plotly_chart(pop_chart)
                    
            else:
                st.error("An error occurred during audio feature retrieval. Please try again.")
        except Exception as e:
            st.error("An error occurred during analysis. Please try again.")
            # st.exception(e)

# Initialize the Spotify client
def init_spotify_client():
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]  #config('SPOTIFY_CLIENT_ID')
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"] # config('SPOTIFY_CLIENT_SECRET')
    credential_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=credential_manager)
    return sp
sp = init_spotify_client()


###### Search Functions for Autocomplete Features ######
def artist_search_func(sp,query) -> List[str]:
    result = sp.search(q=query, type='artist', limit=5)
    artists = [artist['name'] for artist in result['artists']['items']]
    return artists
        
def track_search_func(sp,query) -> List[str]:
    result = sp.search(q=query, type='track', limit=10)
    tracks = [track['name'] for track in result['tracks']['items']]
    return tracks
    
def artist_track_search_func(sp, artist, query) -> List[str]:
    result = sp.search(q=f"artist:{artist} track:{query}", type='track', limit=10)
    tracks = [track['name'] for track in result['tracks']['items']]
    return tracks


#################################
#### application starts here ####
#################################

# Hot Chart playlist options
playlists = {
    "Select a Playlist": None, # Placeholder value
    "Billboard Hot 100": "6UeSakyzhiEt4NB3UAd6NQ",
    "Top 50 Global (Daily)": "37i9dQZEVXbMDoHDwVN2tF",
    "Top Songs Global (Weekly)": "37i9dQZEVXbNG2KDcFcKOF",
    "Big On Ineternet": "37i9dQZF1DX5Vy6DFOcx00",
    "Viral 50 Global (Daily)": "37i9dQZEVXbLiRSasKsNU9"
}

# Sidebar section starts
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
    # Playlist selection
    selected_playlist = st.selectbox("Select a Playlist", options=list(playlists.keys()))
    # Obtain playlist id
    playlist_id = playlists[selected_playlist]
    # Compare button
    compare_button = st.sidebar.button("Compare")


st.write("# Track-To-Chart Comparison")
# Header Text
st.info("You're comparing your tracks with the Latest Top Chart on Spotify, which is updated daily/weekly to reflect the latest top-performing tracks.", icon="👩🏽‍🎤")

# Main section starts
if compare_button and playlist_id and selected_artist and selected_track:
    try: 
        spotify_analyzer = SpotifyAnalyzer(sp, playlist_id)

        st.header('Compare with The Latest Hit Songs')
        st.components.v1.iframe(f"https://open.spotify.com/embed/playlist/{playlist_id}?utm_source=generator&theme=0",
                        width=500, height=160, scrolling=True)
        st.divider()

        # Run the analysis
        spotify_analyzer.run_analysis(selected_artist, selected_track)
        st.balloons()

    except Exception as e:
        print(f'An error occurred: {str(e)}')