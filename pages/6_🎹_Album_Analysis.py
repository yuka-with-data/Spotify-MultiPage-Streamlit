# Load Libraries
import os
import pandas as pd
import numpy as np
import time
import json
import spotipy
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
import warnings

# Suppress specific seaborn warning
warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue` is deprecated")

# Page Config
st.set_page_config(page_title="Album Analysis", 
                   page_icon="🎹")

# Import data_galaxy after Page Config
from data_galaxy import init_spotify_client, retrieve_album_data


# Class
class AlbumAnalyzer:
    def __init__(self, sp, album_id: str) -> None:
        self.sp = sp
        self.mean_values_album, self.df_album = retrieve_album_data(self.sp, album_id)
        self.colorscale = [
            [0.0, "rgba(232, 148, 88, 0.8)"],   # Lighter orange
            [0.12, "rgba(213, 120, 98, 0.8)"],  # Dark orange
            [0.24, "rgba(190, 97, 111, 0.8)"],  # Reddish-pink
            [0.36, "rgba(164, 77, 126, 0.8)"],  # Lighter magenta
            [0.48, "rgba(136, 60, 137, 0.8)"],  # Deep magenta
            [0.58, "rgba(125, 50, 140, 0.8)"],  # Mid purple
            [0.68, "rgba(106, 44, 141, 0.8)"],  # Purple-pink
            [0.78, "rgba(87, 35, 142, 0.8)"],   # Deep purple
            [0.88, "rgba(69, 27, 140, 0.8)"],   # Rich purple
            [0.94, "rgba(40, 16, 137, 0.8)"],   # Darker purple
            [0.97, "rgba(26, 12, 135, 0.8)"],   # Between darker purple and dark blue
            [1.0, "rgba(12, 7, 134, 0.8)"]      # Dark blue
        ]

    def get_color(self, value, min_value, max_value):
        # Ensure that the values are floats
        if isinstance(value, str):
            value = float(value)
        if isinstance(min_value, str):
            min_value = float(min_value)
        if isinstance(max_value, str):
            max_value = float(max_value)
        
        # Normalize the value to a range [0, 1]
        normalized_value = (value - min_value) / (max_value - min_value)
        # Find the corresponding color in the colorscale
        for i in range(len(self.colorscale) - 1):
            if normalized_value <= self.colorscale[i + 1][0]:
                return self.colorscale[i][1]
        return self.colorscale[-1][1]
    

    def radar_chart(self) -> go.Figure:
        color_album = 'rgba(93, 58, 155, 0.9)' 
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
            legend=dict(
                orientation='h',
                x=0.9,
                y=1.1,
            ),
            paper_bgcolor='WhiteSmoke',
            font={"color": "black"},
            height=450,
            width=700,
            margin=dict(l=40, r=40, t=40, b=40),
            autosize=True
        )

        return fig
    
    def tempo_histogram(self) -> go.Figure:
        color_album = px.colors.sequential.Plasma[2]  
        color_average_tempo = px.colors.sequential.Plasma[6]

        # Calculate histogram data
        data_min = self.df_album['tempo'].min()
        data_max = self.df_album['tempo'].max()
        bins = np.linspace(data_min - 0.1, data_max + 0.1, num=50)
        counts, bins = np.histogram(self.df_album['tempo'], bins=bins)
        bins = np.round(bins, 2)

        # Group data into bins
        bin_labels = [f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)-1)]
        self.df_album['bin'] = pd.cut(self.df_album['tempo'], bins=bins, labels=bin_labels, include_lowest=True)

        # Prepare data for tooltips
        grouped = self.df_album.groupby('bin', observed=False)
        print(grouped)
        tooltip_data = grouped['track_name'].agg(lambda x: '<br>'.join(x)).reset_index()

        # Create the figure and add bars
        fig = go.Figure()
        for label, group in grouped:
            fig.add_trace(go.Bar(
                x=[label],
                y=[group['tempo'].count()],
                text=[tooltip_data[tooltip_data['bin'] == label]['track_name'].values[0]],
                hoverinfo="text",
                hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
                marker=dict(color=color_album, line=dict(width=1, color='black')),
                name=label,
                showlegend=False),
                )

        # Calculate mean tempo and find the bin
        mean_tempo = self.df_album['tempo'].mean()
        mean_tempo_bin = pd.cut([mean_tempo], bins=bins, labels=bin_labels, include_lowest=True)[0]

        # Add a line for average tempo
        fig.add_trace(go.Scatter(
            x=[mean_tempo_bin, mean_tempo_bin],
            y=[0, counts.max()], 
            mode='lines',
            line=dict(color=color_average_tempo, width=2, dash='dash'),
            name='Average Tempo',
            hoverinfo='text',
            text=f"Mean Tempo: {mean_tempo:.2f} BPM"
        ))

        # Update layout with additional options
        fig.update_layout(
            xaxis_title='Tempo (BPM)',
            yaxis_title='Frequency',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation='h',
                x=0.8,
                y=1.1,
            ),
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            margin=dict(l=20, r=20, t=20, b=20),
            autosize=True
        )

        return fig


    def duration_histogram(self) -> go.Figure:
        color_album = px.colors.sequential.Plasma[2]
        color_average_duration = px.colors.sequential.Plasma[6]

        # Convert duration from milliseconds to minutes
        self.df_album['duration_min'] = self.df_album['duration_ms'] / 60000

        # Inner function
        def format_min_minsec(minutes):
            full_minutes = int(minutes)
            seconds = int((minutes - full_minutes) * 60)
            return f"{full_minutes}m {seconds}s"
        
        # Calculate histogram data
        data_min = self.df_album['duration_min'].min()
        data_max = self.df_album['duration_min'].max()
        bins = np.linspace(data_min - 0.1, data_max + 0.1, num=50)
        counts, bins = np.histogram(self.df_album['duration_min'], bins=bins)
        bins = np.round(bins, 2)

        # Group data into bins
        # Create bin labels for grouping
        bin_labels = [f"{format_min_minsec(bins[i])} - {format_min_minsec(bins[i+1])}" for i in range(len(bins)-1)]
        self.df_album['bin'] = pd.cut(self.df_album['duration_min'], bins=bins, labels=bin_labels, include_lowest=True)

        # Prepare tooltips
        grouped = self.df_album.groupby('bin', observed=False)
        tooltip_data = grouped['track_name'].agg(lambda x: '<br>'.join(x)).reset_index()

        # Create the figure and add bars
        fig = go.Figure()
        for label, group in grouped:
            tooltip = tooltip_data[tooltip_data['bin'] == label]['track_name'].values[0]
            fig.add_trace(go.Bar(
                x=[label],
                y=[group['duration_min'].count()],
                hoverinfo="text",
                text=[tooltip],
                hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
                marker=dict(color=color_album, line=dict(width=1, color='black')),
                name=label,
                showlegend=False),
                )

        # Calculate mean duration and find the bin
        mean_duration = self.df_album['duration_min'].mean()
        mean_duration_bin = pd.cut([mean_duration], bins=bins, labels=bin_labels, include_lowest=True)[0]

        # Add a line for average duration
        fig.add_trace(go.Scatter(
            x=[mean_duration_bin, mean_duration_bin],
            y=[0, counts.max()],
            mode='lines',
            line=dict(color=color_average_duration, width=2, dash='dash'),
            name='Average Duration',
            hoverinfo='text',
            text=f"Mean Duration: {format_min_minsec(mean_duration)}"
        ))

        # Update layout
        fig.update_layout(
            xaxis_title='Duration (min)',
            yaxis_title='Frequency',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            legend=dict(
                orientation='h',
                x=0.8,
                y=1.1,
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            autosize=True
        )

        return fig
    
    
    def loudness_histogram(self) -> go.Figure:
        color_top_50 = px.colors.sequential.Plasma[2]  
        color_average_loudness = px.colors.sequential.Plasma[6]  

        sorted_df = self.df_album.sort_values(by='loudness', ascending=False)

        fig = go.Figure()

        # Generate color for each bar based on the loudness value
        min_loudness = sorted_df['loudness'].min()
        max_loudness = sorted_df['loudness'].max()
        color_values = sorted_df['loudness'].apply(lambda x: self.get_color(x, min_loudness, max_loudness))

        fig.add_trace(go.Bar(
            x=sorted_df['loudness'],
            y=sorted_df['track_name'],
            orientation='h',
            marker=dict(color=color_values, line=dict(width=1, color='black')),
            text=sorted_df['track_name'],
            hoverinfo='text',
            hovertemplate='<b>%{text}</b><br>Loudness: %{x:.2f} dB<extra></extra>',
            showlegend=False
        ))

        mean_loudness = self.df_album['loudness'].mean()

        # Mean verical line
        fig.add_trace(go.Scatter(
            x=[mean_loudness, mean_loudness],
            y=[sorted_df['track_name'].iloc[-1], sorted_df['track_name'].iloc[0]],  # Full range of y-axis
            mode='lines',
            line=dict(color=color_average_loudness, width=2, dash='dash'),
            name='Average Loudness',
            hoverinfo='text',
            hovertext=[f"Mean Loudness: {mean_loudness:.2f} dB"] * 2,
            showlegend=True
        ))

        fig.update_layout(
            xaxis_title='Loudness (dB)',
            yaxis_title='Track Titles',
            yaxis=dict(autorange='reversed', showticklabels=False),  # Reverse y-axis to have the highest loudness at the top
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            
            legend=dict(
                orientation='h',
                x=0.8,
                y=1.1,
            ),
            height=550,
            width=700,
            margin=dict(l=20, r=20, t=20, b=20),
            autosize=True
        )

        return fig
    
    
    def key_histogram(self) -> go.Figure:
        # Mapping of numeric key values to corresponding alphabetic keys
        key_mapping = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

        grouped = self.df_album.groupby('key', observed=False)['track_name'].apply(list).reindex(range(12), fill_value=[])

        # Get key counts and ensure all keys are present
        key_counts = self.df_album['key'].value_counts().reindex(range(12), fill_value=0)
        
        # Create a DataFrame for easier sorting and mapping
        key_df = pd.DataFrame({
            'Key': range(12),
            'Count': key_counts.values,             
            'Track Names': grouped.values
        })

        # Format track names into a single string per key
        key_df['Formatted Tracks'] = key_df['Track Names'].apply(lambda x: '<br>'.join(x))

        # Map numeric keys to their names
        key_df['Key Name'] = key_df['Key'].apply(lambda x: key_mapping[x])

        # Sort the DataFrame by 'Count'
        key_df_sorted = key_df.sort_values(by='Count', ascending=False)
        # print(key_df_sorted)

        # Map numeric keys to their names
        # key_df_sorted['Key Name'] = key_df_sorted['Key'].apply(lambda x: key_mapping[x])

        fig = go.Figure()

        # Custom Plasma colorscale with 0.8 alpha
        colorscale = [
        [0.0, "rgba(232, 148, 88, 0.8)"],   # Lighter orange
        [0.12, "rgba(213, 120, 98, 0.8)"],  # Dark orange
        [0.24, "rgba(190, 97, 111, 0.8)"],  # Reddish-pink
        [0.36, "rgba(164, 77, 126, 0.8)"],  # Lighter magenta
        [0.48, "rgba(136, 60, 137, 0.8)"],  # Deep magenta
        [0.58, "rgba(125, 50, 140, 0.8)"],  # Mid purple
        [0.68, "rgba(106, 44, 141, 0.8)"],  # Purple-pink
        [0.78, "rgba(87, 35, 142, 0.8)"],   # Deep purple
        [0.88, "rgba(69, 27, 140, 0.8)"],   # Rich purple
        [0.94, "rgba(40, 16, 137, 0.8)"],   # Darker purple
        [0.97, "rgba(26, 12, 135, 0.8)"],   # between darker purple and dark blue
        [1.0, "rgba(12, 7, 134, 0.8)"]      # Dark blue
    ]

        # Add the bar trace with custom tooltips
        fig.add_trace(go.Bar(
            x=key_df_sorted['Key Name'],
            y=key_df_sorted['Count'],
            text=key_df_sorted['Formatted Tracks'],
            hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
            marker=dict(color=key_df_sorted['Count'], colorscale=colorscale)
        ))

        # Update layout
        fig.update_layout(
            # title='Key Distribution',
            xaxis_title='Key',
            yaxis_title='Count',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            autosize=True,
            margin=dict(l=20, r=20, t=30, b=20)
        )

        return fig
    
    def mode_pie_chart(self) -> go.Figure:
        # Prepare data for Major vs. Minor mode
        mode_data = self.df_album[['mode', 'track_name']].copy()
        mode_data['mode'] = mode_data['mode'].map({1: 'Major', 0: 'Minor'})

        # Aggregate track names for each category
        title_summary = mode_data.groupby('mode', observed=False)['track_name'].apply(lambda x: '<br>'.join(x[:5]) + ('...' if len(x) > 5 else '')).reset_index()
        title_summary.columns = ['mode', 'titles']  # Correctly rename columns

        # Calculate counts for each category
        count_summary = mode_data['mode'].value_counts().reset_index()
        count_summary.columns = ['mode', 'count']  # Directly assigning new column names

        # Merge titles and counts data
        final_data = pd.merge(title_summary, count_summary, on='mode')

        # Create the pie chart
        fig = px.pie(
            final_data,
            names='mode',
            values='count',
            color_discrete_sequence=[px.colors.sequential.Plasma[2], px.colors.sequential.Plasma[7]],
            hole=0.2,
            custom_data=['titles']
        )

        # Setting tooltip to display track titles, each on a new line
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<br><b>Tracks:</b><br>%{customdata[0]}<extra></extra>"
        )

        # Update layout
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            autosize=True,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20)
        )

        return fig
    
    def explicit_pie_chart(self) -> go.Figure:
        # Prepare data for Explicit vs. Non-Explicit
        explicit_data = self.df_album[['is_explicit', 'track_name']].copy()
        explicit_data['is_explicit'] = explicit_data['is_explicit'].map({True: 'Explicit', False: 'Non-Explicit'})

        # Aggregate track names for each category
        title_summary = explicit_data.groupby('is_explicit', observed=False)['track_name'].apply(lambda x: '<br>'.join(x[:5]) + ('...' if len(x) > 5 else '')).reset_index()
        title_summary.columns = ['is_explicit', 'titles']

        # Calculate counts for each category
        count_summary = explicit_data['is_explicit'].value_counts().reset_index()
        count_summary.columns = ['is_explicit', 'count'] 

        # Merge titles and counts data
        final_data = pd.merge(title_summary, count_summary, on='is_explicit')

        # Create the pie chart
        fig = px.pie(
            final_data,
            names='is_explicit',
            values='count',
            color_discrete_sequence=[px.colors.sequential.Plasma[2], px.colors.sequential.Plasma[7]],
            hole=0.2,
            custom_data=['titles']
        )

        # Setting tooltip to display track titles, each on a new line
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<br><b>Tracks:</b><br>%{customdata[0]}<extra></extra>"
        )

        # Update layout
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            autosize=True,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20)
        )

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
                {'range': [0, 100], 'color': 'GhostWhite'},
                {'range': [0, mean_popularity], 'color': 'rgba(65, 105, 225, 0.5)'}
                ],
            }
        ))
        fig.update_layout(
            paper_bgcolor='WhiteSmoke',
            template='plotly_white',
            font={'color':"black"},
            height=450,
            width=700,
            autosize=True
        )

        return fig

    
    def run_analysis(self) -> None:
        try:
            st.header('Album DataFrame')
            st.dataframe(self.df_album)

            st.header('Attribute Radar Chart')
            st.text("The radar chart displays various musical attributes of the album to compare their relative strengths.")
            fig = self.radar_chart()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Tempo Histogram')
            st.text("The tempo histogram shows the distribution of the tempo (beats per minute) across all tracks in the album.")
            fig = self.tempo_histogram()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Duration Histogram')
            st.text("The histogram illustrates the distribution of track durations within the album, highlighting variability in song lengths.")
            fig = self.duration_histogram()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Loudness Histogram')
            st.text("The loudness histogram plots the loudness levels (in decibels) of each track, showing the dynamic range of the album.")
            fig = self.loudness_histogram()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Key Histogram')
            st.text("The histogram displays the musical keys of the album's tracks, indicating the most common keys used.")
            fig = self.key_histogram()
            st.plotly_chart(fig, use_container_width=True)

            st.header("Mode Pie Chart")
            st.text("Major modes are bright and uplifting, while minor modes are somber and serious.")
            mode_pie = self.mode_pie_chart()
            st.plotly_chart(mode_pie, use_container_width=True)

            st.header('Explicitness Pie Chart')
            st.text("The pie chart breaks down the proportion of explicit to non-explicit tracks, providing insight into the album's content.")
            fig = self.explicit_pie_chart()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Album Popularity Gauge Chart')
            st.text("The gauge chart displays the album's popularity score, "
                    "which is the average of the popularity scores of all tracks in the album.")
            fig = self.album_popularity_gauge_chart()
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            print(f" Run Analysis Error {e}")
            st.error("An error occurred during analysis. Please try again.")


# Initialize Spotify Client
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

            

