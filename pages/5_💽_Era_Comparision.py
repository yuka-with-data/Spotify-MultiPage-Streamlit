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
from typing import Optional, Dict, Tuple, List, Union, Any


# Page Config
st.set_page_config(page_title="Music Era Comparison", 
                   page_icon="📀")


class EraComparison:
    def __init__(self, sp) -> None:
        self.sp = sp
        self.playlists_ttl = {
        "All Out 50s": ("37i9dQZF1DWSV3Tk4GO2fq", 604800), # 1 week TTL
        "All Out 60s": ("37i9dQZF1DXaKIA8E7WcJj", 604800), 
        "All Out 70s": ("37i9dQZF1DWTJ7xPn4vNaz", 604800),
        "All Out 80s": ("37i9dQZF1DX4UtSsGT1Sbe", 604800),
        "All Out 90s": ("37i9dQZF1DXbTxeAdrVG2l", 604800),
        "All Out 2000s": ("37i9dQZF1DX4o1oenSJRJd", 604800),
        "All Out 2010s": ("37i9dQZF1DX5Ejj0EkURtP", 604800),
        "All Out 2020s": ("37i9dQZF1DX2M1RktxUUHG", 604800)
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


    def compare_playlists(self, playlist_id1:str, playlist_id2:str) -> Tuple[Any, pd.DataFrame, Any, pd.DataFrame]:
        att1, df1 = self.retrieve_latest_data(self.sp, playlist_id1)
        att2, df2 = self.retrieve_latest_data(self.sp, playlist_id2)
        if df1.empty or df2.empty:
            st.error("One of the playlists could not be retrieved successfully.")
            return
        return att1, df1, att2, df2
    

    def get_playlist_name(self, playlist_id:str) -> str:
        # Reverse the mappings in playlist1 and playlist2 so IDs are keys
        reversed_playlists = {v: k for k, v in playlists.items()}

        print("Playlist Map:", reversed_playlists) 
        print("Looking for ID:", playlist_id)

        return reversed_playlists.get(playlist_id, "Unknown Playlist")


    def radar_chart(self, att1, att2, label1, label2) -> go.Figure:
        """ 
        Create a radar chart comparing selected attributes of two eras.
        
        Args:
            att1 (pd.Series): Selected attributes of the first era.
            att2 (pd.Series): Selected attributes of the second era.
            labels (list, optional): Labels for the eras being compared. Defaults to None.
            
        Returns:
            go.Figure
        """
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
            name=label1,
            fillcolor=color_era1,
            line=dict(color=color_era1),
        ))
        
        # Trace for the second era
        fig.add_trace(go.Scatterpolar(
            r=att2_values,
            theta=attributes,
            fill='toself',
            name=label2,
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
            width=700
        )
        
        return fig
    
    def duration_histogram(self, df1, df2, label1, label2) -> plt.Figure:
        """ 
         Create histogram chart comparing track durations between two eras.

         Args:
            df1: DataFrame containing tracks of the first playlist
            df2: DataFrame containing tracks of the second playlist
            label1: Label (name) of the first playlist
            label2: Label (name) of the second playlist
        
         Returns:
            plt.Figure
         """
        fig, axs = plt.subplots(2,1,figsize=(10,8), sharex=True)

        # Convert from milliseconds to seconds
        duration_1 = df1['duration_ms']/1000
        duration_2 = df2['duration_ms']/1000

        mean_1 = duration_1.mean()
        mean_2 = duration_2.mean()

        # Define colors
        color_1 = cm.plasma(0.15)
        color_2 = cm.plasma(0.7)

        # Plot first playlist
        sns.histplot(duration_1,
                     bins=30,
                     kde=True,
                     alpha=0.7,
                     color=color_1,
                     edgecolor='black',
                     label=label1,
                     ax=axs[0])
        axs[0].axvline(mean_1, color='blue', linestyle='dashed', linewidth=2, label=f'Mean Duration: {mean_1:.2f}s')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        axs[0].grid(True, axis='y', linestyle='--',alpha=0.6)
        axs[0].set_facecolor('whitesmoke')
        
        # Plot second playlist
        sns.histplot(duration_2,
                     bins=30,
                     kde=True,
                     alpha=0.7,
                     color=color_2,
                     edgecolor='black',
                     label=label2,
                     ax=axs[1])
        axs[1].axvline(mean_2, color='blue', linestyle='dashed', linewidth=2, label=f'Mean Duration: {mean_2:.2f}s')
        axs[1].set_xlabel('Duration (in seconds)')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        axs[1].grid(True, axis='y', linestyle='--', alpha=0.6)
        axs[1].set_facecolor('whitesmoke')

        # Ensure y-axis ticks are int
        axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))

        fig.patch.set_facecolor('lightgrey')
        fig.tight_layout(pad=3.0)

        return fig
    

    def tempo_histogram(self, df1, df2, label1, label2) -> plt.Figure:
        """ 
         Create histogram chart comparing track tempo between two eras.

         Args:
            df1: DataFrame containing tracks of the first playlist
            df2: DataFrame containing tracks of the second playlist
            label1: Label (name) of the first playlist
            label2: Label (name) of the second playlist
        
         Returns:
            plt.Figure
         """
        fig, axs = plt.subplots(2,1,figsize=(10,8), sharex=True)

        tempo_1 = df1['tempo']
        tempo_2 = df2['tempo']

        mean1 = tempo_1.mean()
        mean2 = tempo_2.mean()

        # Define colors
        color_1 = cm.plasma(0.15)
        color_2 = cm.plasma(0.7)

        # Plot first playlist
        sns.histplot(tempo_1,
                     bins=30,
                     kde=True,
                     alpha=0.7,
                     color=color_1,
                     edgecolor='black',
                     label=label1,
                     ax=axs[0])
        axs[0].axvline(mean1, color='blue', linestyle='dashed', linewidth=2, label=f'Mean Tempo: {mean1:.2f} BPM')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        axs[0].grid(True, axis='y', linestyle='--',alpha=0.6)
        axs[0].set_facecolor('whitesmoke')
        
        # Plot second playlist
        sns.histplot(tempo_2,
                     bins=30,
                     kde=True,
                     alpha=0.7,
                     color=color_2,
                     edgecolor='black',
                     label=label2,
                     ax=axs[1])
        axs[1].axvline(mean2, color='blue', linestyle='dashed', linewidth=2, label=f'Mean Tempo: {mean2:.2f} BPM')
        axs[1].set_xlabel('Tempo (BPM)')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        axs[1].grid(True, axis='y', linestyle='--', alpha=0.6)
        axs[1].set_facecolor('whitesmoke')

        # Ensure y-axis ticks are int
        axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))

        fig.patch.set_facecolor('lightgrey')
        fig.tight_layout(pad=3.0)

        return fig
    
    def key_distribution(self, df1, df2, label1, label2) -> plt.Figure:
        """
        Compare the key distribution between two playlists.
        
        Args:
            df1: DataFrame for the first playlist.
            df2: DataFrame for the second playlist.
            label1: Label for the first playlist.
            label2: Label for the second playlist.
            
        Returns:
            plt.Figure
        """
        # Key mapping
        key_mapping = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

        # Calculating key distribution and sorting
        key_counts_1 = df1['key'].value_counts().reindex(range(12), fill_value=0).sort_index()
        key_counts_2 = df2['key'].value_counts().reindex(range(12), fill_value=0).sort_index()

        fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Plot for first DataFrame
        # Specify x-axis value. 
        # Apply the mapping to each index to convert each numeric to alphabetic key
        bars1 = axs[0].bar(key_counts_1.index.map(lambda x: key_mapping[x]), key_counts_1.values)
        axs[0].set_title(label1)
        axs[0].set_xlabel('Key')
        axs[0].set_ylabel('Count')
        axs[0].set_facecolor('whitesmoke')

        # Color the bars using the plasma colormap
        # generates a sequence of colors using the "plasma" colormap
        # creates a linear array of numbers from 0 to 1, inclusive
        # Passing this array to generates colors from the plasma colormap
        # pairs each bar in bars1 with a color from the generated sequence of colors
        # zip creates tuples where the first element is a bar and the second element is a color
        num_bars = len(key_counts_1)
        for bar, color in zip(bars1, plt.cm.plasma(np.linspace(0, 1, num_bars))):
            rgba_color1 = (*color[:3], 0.8)
            bar.set_color(rgba_color1)

        # Plot for second DataFrame
        bars2 = axs[1].bar(key_counts_2.index.map(lambda x: key_mapping[x]), key_counts_2.values)
        axs[1].set_title(label2)
        axs[1].set_xlabel('Key')
        axs[1].set_facecolor('whitesmoke')

        # Color the bars using the plasma colormap
        for bar, color in zip(bars2, plt.cm.plasma(np.linspace(0, 1, num_bars))):
            rgba_color2 = (*color[:3], 0.8)
            bar.set_color(rgba_color2)

        fig.patch.set_facecolor('lightgrey')
        fig.tight_layout(pad=3.0)
        return fig


    def loudness_histogram(self, df1, df2, label1, label2) -> plt.Figure:
            """ 
            Create histogram chart comparing track loudness between two eras.

            Args:
                df1: DataFrame containing tracks of the first playlist
                df2: DataFrame containing tracks of the second playlist
                label1: Label (name) of the first playlist
                label2: Label (name) of the second playlist
            
            Returns:
                plt.Figure
            """
            fig, axs = plt.subplots(2,1,figsize=(10,8), sharex=True)

            loudness_1 = df1['loudness']
            loudness_2 = df2['loudness']

            mean1 = loudness_1.mean()
            mean2 = loudness_2.mean()

            # Define colors
            color_1 = cm.plasma(0.15)
            color_2 = cm.plasma(0.7)

            # Plot first playlist
            sns.histplot(loudness_1,
                        bins=30,
                        kde=True,
                        alpha=0.7,
                        color=color_1,
                        edgecolor='black',
                        label=label1,
                        ax=axs[0])
            axs[0].axvline(mean1, color='blue', linestyle='dashed', linewidth=2, label=f'Mean loudness: {mean1:.2f} BPM')
            axs[0].set_ylabel('Frequency')
            axs[0].legend()
            axs[0].grid(True, axis='y', linestyle='--',alpha=0.6)
            axs[0].set_facecolor('whitesmoke')
            
            # Plot second playlist
            sns.histplot(loudness_2,
                        bins=30,
                        kde=True,
                        alpha=0.7,
                        color=color_2,
                        edgecolor='black',
                        label=label2,
                        ax=axs[1])
            axs[1].axvline(mean2, color='blue', linestyle='dashed', linewidth=2, label=f'Mean loudness: {mean2:.2f} BPM')
            axs[1].set_xlabel('Loudness (dB)')
            axs[1].set_ylabel('Frequency')
            axs[1].legend()
            axs[1].grid(True, axis='y', linestyle='--', alpha=0.6)
            axs[1].set_facecolor('whitesmoke')

            # Ensure y-axis ticks are int
            axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
            axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))

            fig.patch.set_facecolor('lightgrey')
            fig.tight_layout(pad=3.0)

            return fig
    
    def genre_wordcloud(self, df1, df2, label1, label2) -> plt.Figure:
        """
        Create Word Clouds displaying the genre overview for two different playlists 
        and display them side by side or top and bottom.
        
        Args:
        df1: DataFrame for the first playlist.
        df2: DataFrame for the second playlist.
        title1: Title for the first playlist's word cloud.
        title2: Title for the second playlist's word cloud.
        
        Returns:
            plt.Figure
        """
        def clean_genres(genres_str) -> plt.Figure:
            # inner function to clean the genre str
            genres_list = genres_str.split(',')
            cleaned_genres = ', '.join([genre.strip() for genre in genres_list if genre.strip() != ''])
            return cleaned_genres
        
        all_genres_1 = ', '.join(df1['genres'].apply(clean_genres))
        all_genres_2 = ', '.join(df2['genres'].apply(clean_genres))

        regexp = r"\w(?:[-']?\w)+"
        wc1 = WordCloud(width=700, 
                        height=500, 
                        background_color='whitesmoke', 
                        colormap='plasma_r', 
                        regexp=regexp, 
                        scale=2, 
                        max_words=200,
                        min_font_size=10).generate(all_genres_1)
        
        wc2 = WordCloud(width=700, 
                        height=500, 
                        background_color='whitesmoke', 
                        colormap='plasma_r', 
                        regexp=regexp, 
                        scale=2, 
                        max_words=200,
                        min_font_size=10).generate(all_genres_2)

        fig, axs = plt.subplots(2, 1, figsize=(14,7))
        axs[0].imshow(wc1, interpolation='bilinear')
        axs[0].axis("off")
        axs[0].set_title(label1)

        axs[1].imshow(wc2, interpolation='bilinear')
        axs[1].axis("off")
        axs[1].set_title(label2)

        plt.tight_layout(pad=0)
        return fig
    
    def explicit_pie_chart(self, df1, df2, pl1, pl2) -> plt.Figure:
        """ 
         Creates side-by-side pie charts comparing the explicit content percentages 
         between two playlists.

         Args:
          df1 : pandas.DataFrame
          df2 : pandas.DataFrame
          pl1 : str
          pl2 : str

         Returns:
          plt.Figure
           """
        def get_explicit_data(df):
            explicit_count = df['is_explicit'].sum()
            non_explicit_count = len(df) - explicit_count
            return [explicit_count, non_explicit_count]
        
        # Get explicit data from each playlist (in the list)
        explicit_1 = get_explicit_data(df1)
        explicit_2 = get_explicit_data(df2)

        labels = ['Explicit', 'Non Explicit']
        colors = [cm.plasma(0.10, alpha=0.75), cm.plasma(0.65, alpha=0.75)]

        # Initialize subplots for side-by-side pie charts
        fig, axs = plt.subplots(1,2,figsize=(12,6))

        axs[0].pie(explicit_1,
                   labels=labels,
                   colors=colors,
                   autopct='%1.1f%%',
                   startangle=90,
                   textprops={'fontsize': 12}, 
                   radius=1.2)
        axs[0].set_title(pl1)
        axs[0].axis('equal')
        
        axs[1].pie(explicit_2,
                   labels=labels,
                   colors=colors,
                   autopct='%1.1f%%',
                   startangle=90,
                   textprops={'fontsize': 12}, 
                   radius=1.2)
        axs[1].set_title(pl2)
        axs[1].axis('equal')

        fig.patch.set_facecolor('lightgrey')
        plt.tight_layout()

        return fig


    def run_analysis(self, id1, id2):
        att1, df1, att2, df2 = self.compare_playlists(id1, id2)
        label1 = self.get_playlist_name(id1)  # Placeholder function
        label2 = self.get_playlist_name(id2) 
        
        st.header('Radar Chart Comparison:')
        st.text("Music Era Comparison of Attributes (Mean Values)")
        radarchart = self.radar_chart(att1, att2, label1, label2)
        st.plotly_chart(radarchart)

        st.header('Duration Histogram Comparison:')
        st.text("Music Era Comparison of Track Duration")
        durationhist = self.duration_histogram(df1, df2, label1, label2)
        st.pyplot(durationhist)

        st.header('Tempo (BPM) Histogram Comparision:')
        st.text("Music Era Comparision of Tempo")
        tempohist = self.tempo_histogram(df1, df2, label1, label2)
        st.pyplot(tempohist)

        st.header('Key Distribution Comparision:')
        st.text("Music Era Comparision of Key")
        keybar = self.key_distribution(df1, df2, label1, label2)
        st.pyplot(keybar)

        st.header('Loudness (dB) Histogram Comparison:')
        st.text("Music Era Comparision of Loudness")
        loudhist = self.loudness_histogram(df1, df2, label1, label2)
        st.pyplot(loudhist)

        st.header('Artist Genres Word Cloud Comparison:')
        st.text("Music Era Comparison of Word Cloud")
        wcfig = self.genre_wordcloud(df1, df2, label1, label2)
        if wcfig is not None:
            st.pyplot(wcfig)
        else:
            st.error("Word Cloud could not be generated due to the insufficient data.")

        st.header('Explicitness Pie Chart Comparision:')
        st.text("Music Era Comparision of Explicitness")
        explicit = self.explicit_pie_chart(df1, df2, label1, label2)
        st.pyplot(explicit)
        

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

playlists = {
    "Select a Playlist": None, # Placeholder value
    "All Out 50s": "37i9dQZF1DWSV3Tk4GO2fq", 
    "All Out 60s": "37i9dQZF1DXaKIA8E7WcJj", 
    "All Out 70s": "37i9dQZF1DWTJ7xPn4vNaz",
    "All Out 80s": "37i9dQZF1DX4UtSsGT1Sbe",
    "All Out 90s": "37i9dQZF1DXbTxeAdrVG2l",
    "All Out 2000s": "37i9dQZF1DX4o1oenSJRJd",
    "All Out 2010s": "37i9dQZF1DX5Ejj0EkURtP",
    "All Out 2020s": "37i9dQZF1DX2M1RktxUUHG"
}

with st.sidebar:
    st.title("Select Playlist By Era")

    # Playlist selection
    selected_playlist1 = st.selectbox("Select Era 1", options=list(playlists.keys()), key='playlist1_selectbox')
    selected_playlist2 = st.selectbox("Select Era 2", options=list(playlists.keys()), key='playlist2_selectbox')

    # Obtain playlist id
    playlist_id1 = playlists[selected_playlist1]
    playlist_id2 = playlists[selected_playlist2]

    # Compare button
    compare_button = st.sidebar.button("Compare")

# main
st.write("# Music Era Comparison")
# Header text
st.info("You're comparing...", icon="👩🏽‍🎤")

if compare_button and playlist_id1 and playlist_id2:
    try:
        era_comparision = EraComparison(sp)

        st.header('Compare 2 Different Eras of Music')
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
