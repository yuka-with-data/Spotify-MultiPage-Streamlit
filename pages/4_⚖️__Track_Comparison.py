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
st.set_page_config(page_title="Track-To-Chart Comparison", 
                   page_icon="⚖️")

# Import data_galaxy after Page Config
from data_galaxy import init_spotify_client, retrieve_playlist_data

# Class
class SpotifyAnalyzer:
    def __init__(self, sp, playlist_id:str) -> None:
        self.sp = sp
        # Load Playlist Data
        self.mean_values_top_50, self.df_top_50 = retrieve_playlist_data(self.sp, playlist_id)
        self.colorscale = [
            [0.0, "rgba(12, 7, 134, 1.0)"],       # Dark blue
            [0.12, "rgba(26, 12, 135, 1.0)"],     # Between darker purple and dark blue
            [0.24, "rgba(40, 16, 137, 1.0)"],     # Darker purple
            [0.36, "rgba(69, 27, 140, 1.0)"],     # Rich purple
            [0.48, "rgba(87, 35, 142, 1.0)"],     # Deep purple
            [0.58, "rgba(106, 44, 141, 1.0)"],    # Purple-pink
            [0.68, "rgba(125, 50, 140, 1.0)"],    # Mid purple
            [0.78, "rgba(136, 60, 137, 1.0)"],    # Deep magenta
            [0.88, "rgba(164, 77, 126, 1.0)"],    # Lighter magenta
            [0.94, "rgba(190, 97, 111, 1.0)"],    # Reddish-pink
            [0.97, "rgba(213, 120, 98, 1.0)"],    # Dark orange
            [1.0, "rgba(232, 148, 88, 1.0)"]      # Lighter orange
        ]

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

        color_top_50 = "rgba(69, 27, 140, 0.9)" # Rich Purple
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
            name=f'{track_name}',
            fillcolor=color_your_track,
            line=dict(color=color_your_track, dash='dot'),
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
    
    def create_bpm_histogram(self, audio_features: Dict[str, float], user_track_name: str) -> go.Figure:
        """ 
         Create a histogram chart comparing BPM(tempo) of a track with the top 100
         Args:
           audio_features (dict)
         Returns:
           histogram chart
           """
        # Use Plasma colormap colors
        color_top_50 = "rgba(69, 27, 140, 0.9)" # Rich Purple
        color_your_track = px.colors.sequential.Plasma[4]  
        color_average_tempo = px.colors.sequential.Plasma[6]   

        # Calculate histogram data
        data_min = self.df_top_50['tempo'].min()
        data_max = self.df_top_50['tempo'].max()
        bins = np.linspace(data_min - 0.1, data_max + 0.1, num=50)
        counts, bins = np.histogram(self.df_top_50['tempo'], bins=bins)
        bins = np.round(bins, 2)

        # Group data into bins
        bin_labels = [f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)-1)]
        self.df_top_50['bin'] = pd.cut(self.df_top_50['tempo'], bins=bins, labels=bin_labels, include_lowest=True)
        
        # Prepare data for the tooltips
        grouped = self.df_top_50.groupby('bin', observed=False)
        tooltip_data = grouped['track_name'].agg(lambda x: '<br>'.join(x)).reset_index()

        fig = go.Figure()
        for label, group in grouped:
            fig.add_trace(go.Bar(
                x=[label],
                y=[group['tempo'].count()],
                text=[tooltip_data[tooltip_data['bin'] == label]['track_name'].values[0]],
                hoverinfo="text",
                hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
                marker=dict(color=color_top_50, line=dict(width=1, color='black')),
                name=label,
                showlegend=False),
                )

        # Calculate mean tempo and find the bin for the mean
        mean_tempo = self.df_top_50['tempo'].mean()
        mean_tempo_bin = pd.cut([mean_tempo], bins=bins, labels=bin_labels, include_lowest=True)[0]

        # Add line for the average tempo
        fig.add_trace(go.Scatter(
            x=[mean_tempo_bin, mean_tempo_bin],
            y=[0, counts.max()],
            mode='lines',
            line=dict(color=color_average_tempo, width=2, dash='dash'),
            name='Average Tempo',
            hoverinfo='text',
            text=f"Mean Tempo: {mean_tempo:.2f} BPM" 
        ))

        # Find the bin for user's track tempo and add a vertical line
        user_tempo = audio_features['tempo'] 
        user_tempo_bin = pd.cut([user_tempo], bins=bins, labels=bin_labels, include_lowest=True)[0]
        fig.add_trace(go.Scatter(
            x=[user_tempo_bin, user_tempo_bin],
            y=[0, counts.max()],
            mode='lines',
            line=dict(color=color_your_track, width=2),
            name=f"{user_track_name}",
            hoverinfo='text',
            text=f"{user_track_name}: {user_tempo:.2f} BPM"
        ))


        # Update layout
        fig.update_layout(
            xaxis_title='Tempo (BPM)',
            yaxis_title='Frequency',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            legend=dict(
                orientation='h',
                y=1.1
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            autosize=True
        )

        return fig

    def create_key_distribution_chart(self, audio_features:Dict[str,float], track_name: str) -> go.Figure:
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
        key_counts = self.df_top_50['key'].value_counts().reindex(range(12), fill_value=0).sort_values(ascending=False)
        print(f"Key Counts Ordered: {key_counts}")

        # User's selected key
        user_key_index = audio_features['key']

        fig = go.Figure()

        # Add bars to the figure
        for i, count in enumerate(key_counts):
            # check if current bar is the user's selected key
            opacity = 1.0 if i == user_key_index else 0.3
            # Set hovertemplate only for the selected key
            if i == user_key_index:
                hovertemplate=f'{track_name}</b><extra></extra>',
                hoverinfo='y+name'
            else:
                hovertemplate=''
                hoverinfo='skip'

            fig.add_trace(go.Bar(
                x=[key_mapping[i]],
                y=[count],
                marker=dict(
                    # if there are more bars than colors, the color selection starts again from beginning
                    color=self.colorscale[i % len(self.colorscale)][1], # access the second element of each tuple in colorscale
                    opacity=opacity
                ),
                hoverinfo=hoverinfo,
                hovertemplate=hovertemplate
            ))
        
        # Update layout
        fig.update_layout(
            xaxis_title="Key",
            yaxis_title="Frequency",
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
            autosize=True
        )

        return fig

    
    def create_duration_histogram(self, audio_features: Dict[str, float], user_track_name:str) -> go.Figure:
        """ 
        Create a histogram chart comparing BPM(tempo) of a track with the top 100
        Args:
        audio_features (dict)
        Returns:
        histogram chart
        """
        def format_min_to_minsec(minutes):
            # Convert decimal minutes to minutes and seconds format
            full_minutes = int(minutes)
            seconds = int((minutes - full_minutes) * 60)
            return f"{full_minutes}m {seconds}s"
        
        # Set colors
        color_top_50 = "rgba(69, 27, 140, 0.9)" # Rich Purple 
        color_your_track = px.colors.sequential.Plasma[4]  
        color_average_duration = px.colors.sequential.Plasma[6] 

        # Convert duration from milliseconds to seconds
        self.df_top_50['duration_min'] = self.df_top_50['duration_ms'] / 60000
        audio_duration_min = audio_features['duration_ms'] / 60000

        # Calculate histogram data
        data_min = self.df_top_50['duration_min'].min()
        data_max = self.df_top_50['duration_min'].max()
        bins = np.linspace(data_min - 0.1, data_max + 0.1, num=50)
        counts, bins = np.histogram(self.df_top_50['duration_min'], bins=bins)
        bins = np.round(bins, 2)

        # Group data into bins
        bin_labels = [f"{format_min_to_minsec(bins[i])} - {format_min_to_minsec(bins[i+1])}" for i in range(len(bins)-1)]
        self.df_top_50['bin'] = pd.cut(self.df_top_50['duration_min'], bins=bins, labels=bin_labels, include_lowest=True)
        tooltip_data = self.df_top_50.groupby('bin', observed=False)['track_name'].agg(lambda x: '<br>'.join(x)).reset_index()

        fig = go.Figure()
        for label, group in self.df_top_50.groupby('bin', observed=False):
            fig.add_trace(go.Bar(
                x=[label],
                y=[group['duration_min'].count()],
                text=[tooltip_data[tooltip_data['bin'] == label]['track_name'].values[0]],
                hoverinfo="text",
                hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
                marker=dict(color=color_top_50, line=dict(width=1, color='black')),
                name=label,
                showlegend=False),
                )

        # Calculate mean duration and find the bin for the mean
        mean_duration_min = self.df_top_50['duration_min'].mean()
        mean_bin_index = np.digitize([mean_duration_min], bins)[0] - 1
        mean_bin_label = bin_labels[mean_bin_index] if mean_bin_index < len(bin_labels) else bin_labels[-1]
        
        fig.add_trace(go.Scatter(
            x=[mean_bin_label, mean_bin_label],
            y=[0, counts.max()],
            mode='lines',
            line=dict(color=color_average_duration, width=2, dash='dash'),
            name='Average Duration',
            hoverinfo='text',
            text=f"Mean Duration: {format_min_to_minsec(mean_duration_min)}"
        ))

        # Find the bin for user's track duration and add a vertical line
        user_bin_index = np.digitize([audio_duration_min], bins)[0] - 1
        user_bin_label = bin_labels[user_bin_index] if user_bin_index < len(bin_labels) else bin_labels[-1]
        fig.add_trace(go.Scatter(
            x=[user_bin_label, user_bin_label],
            y=[0, counts.max()],
            mode='lines',
            line=dict(color=color_your_track, width=2),
            name=f"{user_track_name}",
            hoverinfo='text',
            text=f"{user_track_name}: {format_min_to_minsec(audio_duration_min)}"
        ))

        # Update layout
        fig.update_layout(
            xaxis_title='Duration (minutes)',
            yaxis_title='Frequency',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            legend=dict(
                orientation='h',
                y=1.1
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            autosize=True
        )

        return fig


    def create_loudness_histogram(self, audio_features: Dict[str, float], user_track_name:str) -> plt.Figure:
        """ 
        Create a histogram chart comparing loudness of a track with the top 50
        Args:
        audio_features (dict)
        Returns:
        histogram chart
        """
        # Define colors using Plotly's Plasma colormap
        color_top_50 = "rgba(69, 27, 140, 0.9)" # Rich Purple 
        color_your_track = px.colors.sequential.Plasma[4]
        color_average_loudness = px.colors.sequential.Plasma[6]

        # Calculate Histogram data
        data_min = self.df_top_50['loudness'].min()
        data_max = self.df_top_50['loudness'].max()
        bins = np.linspace(data_min - 0.1, data_max + 0.1, num=50)
        counts, bins = np.histogram(self.df_top_50['loudness'], bins=bins)
        bins = np.round(bins, 2)

        # Group data into bins
        bin_labels = [f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)-1)]
        self.df_top_50['bin'] = pd.cut(self.df_top_50['loudness'], bins=bins, labels=bin_labels, include_lowest=True)

        # Prepare data for the tooltips
        grouped = self.df_top_50.groupby('bin', observed=False)
        tooltip_data = grouped['track_name'].agg(lambda x: '<br>'.join(x)).reset_index()
        
        fig = go.Figure()

        # Add bars for histogram
        for label, group in grouped:
            fig.add_trace(go.Bar(
                x=[label],
                y=[group['tempo'].count()],
                text=[tooltip_data[tooltip_data['bin'] == label]['track_name'].values[0]],
                hoverinfo="text",
                hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
                marker=dict(color=color_top_50, line=dict(width=1, color='black')),
                name=label,
                showlegend=False),
                )

        # Add vertical line for average loudness
        mean_loudness = self.df_top_50['loudness'].mean()
        mean_loudness_bin = pd.cut([mean_loudness], bins=bins, labels=bin_labels, include_lowest=True)[0]
        fig.add_trace(go.Scatter(
            x=[mean_loudness_bin, mean_loudness_bin],
            y=[0, counts.max()],
            mode='lines',
            line=dict(color=color_average_loudness, width=2, dash='dash'),
            name='Average Loudness',
            hoverinfo='text',
            text=f"Mean Loudness: {mean_loudness} dB"
        ))

        # Add vertical line for the user's track loudness
        user_loudness = audio_features['loudness']
        user_bin = pd.cut([user_loudness], bins=bins, labels=bin_labels, include_lowest=True)[0]
        fig.add_trace(go.Scatter(
            x=[user_bin, user_bin],
            y=[0, counts.max()],
            mode='lines',
            line=dict(color=color_your_track, width=2),
            name=f'{user_track_name}',
            hoverinfo='text',
            text=f"{user_track_name}: {user_loudness} dB"
        ))

        # Update layout
        fig.update_layout(
            xaxis_title='Loudness (dB)',
            yaxis_title='Frequency',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            legend=dict(
                orientation='h',
                y=1.1,
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )

        return fig


    def create_explicit_pie_chart(self, audio_features: Dict[str, float], user_track_name:str) -> plt.Figure:
        """
        Create a pie chart displaying the percentage of explicit tracks in the top 50
        Returns:
        pie chart
        """

        explicit_color = "rgba(26, 12, 135, 0.8)"
        nonexplicit_color = "rgba(213, 120, 98, 0.8)"

        # Extract the attribute from user's selected track 
        is_explicit = audio_features.get('is_explicit', False)

        # Count values for each category
        explicit_counts = self.df_top_50['is_explicit'].sum()
        non_explicit_count = len(self.df_top_50) - explicit_counts

        counts = [explicit_counts, non_explicit_count]
        labels = ['Explicit', 'Non-Explicit']
        colors = [explicit_color, nonexplicit_color]  # Use Plotly's Plasma colors

        fig = go.Figure(go.Pie(
            labels=labels,
            values=counts,
            pull=[0.02 if is_explicit else 0, 0 if is_explicit else 0.02],  # pull out slice for visual emphasis
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='label+percent',
            insidetextorientation='horizontal'  # adjust the text orientation inside slices
        ))

        # Customize hover information
        fig.update_traces(hoverinfo='label+percent', textinfo='label+percent')

        # Convert the boolean to a readable string
        explicit_status = "Explicit" if is_explicit else "Non-Explicit"

        # Update layout for a clean look
        fig.update_layout(
            title_text=f"Track Explicitness for {user_track_name}: '{explicit_status}'",
            title_font=dict(family='Roboto'),
            title_x=0,
            title_y=0.98,
            showlegend=False,
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='WhiteSmoke',  
            plot_bgcolor='WhiteSmoke', 
            autosize=True
        )

        return fig
    
    def create_mode_pie_chart(self, audio_features: Dict[str, float], user_track_name: str) -> go.Figure:
        """
        Create a pie chart displaying the percentage of tracks in Major and Minor modes in the top chart
        and highlight the mode of the selected track.
        Returns:
        pie chart
        """
        # Set colors
        major_color = "rgba(26, 12, 135, 0.8)"
        minor_color = "rgba(213, 120, 98, 0.8)"

        # Extract attribute from user's selected track
        user_mode = audio_features.get('mode', 1) # boolean

        mode_counts = self.df_top_50['mode'].value_counts()
        major_count = mode_counts.get(1,0)
        minor_count = mode_counts.get(0,0)

        counts = [major_count, minor_count]
        labels = ['Major', 'Minor']
        colors = [major_color, minor_color]

        fig = go.Figure(go.Pie(
            labels=labels, 
            values=counts,
            pull=[0.02 if user_mode else 0, 0 if user_mode else 0.02],
            marker=dict(
                colors=colors,
                line=dict(
                    color='white',
                    width=2
                )
            ),
            textinfo='label+percent',
            insidetextorientation='horizontal'
        ))

        # Customize hover info
        fig.update_traces(hoverinfo='label+percent',
                          textinfo='label+percent')
        
        # Convert the mode boolean to string
        mode_status = "Major" if user_mode else "Minor"

        # Update layout 
        fig.update_layout(
            title_text=f"Mode for {user_track_name}: '{mode_status}'",
            title_font=dict(family='Roboto'),
            title_x=0,  # Align the title to the left
            title_y=0.98,  # Adjust the vertical position of the title
            showlegend=False,
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='WhiteSmoke',
            plot_bgcolor='WhiteSmoke',
            autosize=True
        )
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
            # cleand_genres = [genre.strip() for genre in genres_list if genre.strip() != '']
            cleaned_genres = []
            for genre in genres_list:
                stripped = genre.strip()
                if stripped != '':
                    cleaned_genres.append(stripped)
            cleaned_genres = ', '.join(cleaned_genres)
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
                              background_color='#F5F5F5',
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
                        {'range': [0, 100], 'color': 'GhostWhite'},
                        {'range': [0, popularity], 'color': 'rgba(26, 12, 135, 0.5)'}
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
                paper_bgcolor="WhiteSmoke",
                font={"color": "black"},
                height=450,
                width=700,
                autosize=True
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
                st.plotly_chart(fig, use_container_width=True)

                # Create a BPM Histogram Chart
                st.header('Histogram Chart Comparison - Tempo:')
                st.text(f'Comparison of Tempo in % {selected_playlist} vs. {track_name} by {artist_name}')
                bmp_hist_chart = self.create_bpm_histogram(audio_features, track_name)
                st.plotly_chart(bmp_hist_chart, use_container_width=True)

                # Show Key Distribution
                st.header('Key Distribution Comparison:')
                st.text(f'Distribution of Songs by Key: {selected_playlist} vs. {track_name} by {artist_name}')
                key_dist = self.create_key_distribution_chart(audio_features, track_name)
                st.plotly_chart(key_dist, use_container_width=True)

                # Create a Duration Histogram Chart
                st.header('Histogram Chart Comparison - Duration:')
                st.text(f'Comparison of Duration {selected_playlist} vs. {track_name} by {artist_name}')
                duration_chart = self.create_duration_histogram(audio_features, track_name)
                st.plotly_chart(duration_chart, use_container_width=True)

                # Create a Loudness Histogram Chart
                st.header('Histogram Chart Comparison - Loudness:')
                st.text(f'Comparison of Loudness {selected_playlist} vs. {track_name} by {artist_name}')
                loudness_chart = self.create_loudness_histogram(audio_features, track_name)
                st.plotly_chart(loudness_chart, use_container_width=True)

                # Create a Mode Pie Chart
                st.header('Mode Pie Chart:')
                st.text(f'Comparison of Mode for {selected_playlist} & {track_name} by {artist_name}')
                mode_chart = self.create_mode_pie_chart(audio_features, track_name)
                st.plotly_chart(mode_chart, use_container_width=True)

                # Create a Explicit Pie Chart
                st.header('Explicitness Pie Chart:')
                st.text(f'Comparison of Explicitness {selected_playlist} & {track_name} by {artist_name}')
                explicit_chart = self.create_explicit_pie_chart(audio_features, track_name)
                st.plotly_chart(explicit_chart, use_container_width=True)

                # Create a Genres Word Cloud
                st.header("Genres Word Cloud:")
                st.text(f'Genres Word Cloud for {selected_playlist}')
                user_genres = audio_features.get('genres', '')
                if user_genres:
                    st.write(f"<p style='font-size:24px; font-family:Roboto;'>{selected_artist}'s Genres: {user_genres}</p>", unsafe_allow_html=True)
                else:
                    st.warning("No genres data available for the selected track/artist")
                fig = self.create_genres_wordcloud()
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.error("Word Cloud could not be generated due to the insufficient data.")
                

                # Create a Popularity Gauge Chart
                st.header('Popularity Gauge Chart:')
                st.text(f'Current Popularity Score for {track_name} by {artist_name}')
                pop_chart = self.create_popularity_chart(audio_features)
                st.plotly_chart(pop_chart, use_container_width=True)
                    
            else:
                st.error("An error occurred during audio feature retrieval. Please try again.")
        except Exception as e:
            st.error("An error occurred during analysis. Please try again.")
            # st.exception(e)


# Initialize the Spotify client
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
    "Viral 50 Global (Daily)": "37i9dQZEVXbLiRSasKsNU9",
    "TikTok Top 50 (Weekly)": "4FLeoROn5GT7n2tZq5XB4V",
    "TikTok Viral Hits (Weekly)": "6mKEyAOZ82zQm4ysV3LvqQ"
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