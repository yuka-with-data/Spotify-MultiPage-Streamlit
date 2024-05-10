# Load Libraries
import pandas as pd
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# Import data_galaxy after Page Config
from data_galaxy import init_spotify_client, retrieve_era_data


class EraComparison:
    def __init__(self, sp) -> None:
        self.sp = sp

    def retrieve_latest_data(self, playlist_id: str)-> Tuple[pd.Series, pd.DataFrame]:
        try:
            data_series, data_frame = retrieve_era_data(self.sp, playlist_id)
            if data_frame.empty:
                st.warning("Data retrieval returned an empty DataFrame. There might be no data available or an error occurred.")
                return pd.Series(), pd.DataFrame()
            return data_series, data_frame
        except Exception as e:
            st.error(f"An error occurred while retrieving data: {str(e)}")
            return pd.Series(), pd.DataFrame()
        
    def compare_playlists(self, playlist_id1:str, playlist_id2:str) -> Tuple[Any, pd.DataFrame, Any, pd.DataFrame]:
        att1, df1 = self.retrieve_latest_data(playlist_id1)
        att2, df2 = self.retrieve_latest_data(playlist_id2)
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
            legend=dict(
                orientation='h',
                x=0.7,
                y=1.1,
            ),
            paper_bgcolor='WhiteSmoke',
            font={"color": "black"},
            margin=dict(l=40, r=40, t=40, b=40),
            height=450,
            width=700,
            autosize=True
        )
        
        return fig
    
    def duration_histogram(self, df1, df2, label1, label2) -> go.Figure:
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
        def format_min_minsec(minutes):
            full_minutes = int(minutes)
            seconds = int((minutes - full_minutes) * 60)
            return f"{full_minutes}m {seconds}s"

        color_1 = px.colors.sequential.Plasma[2]
        color_2 = px.colors.sequential.Plasma[6]

        # Convert duration from milliseconds to minutes for readability
        df1['duration_min'] = df1['duration_ms'] / 60000
        df2['duration_min'] = df2['duration_ms'] / 60000

        # Define bins for histogram
        bins = np.linspace(min(df1['duration_min'].min(), df2['duration_min'].min()),
                        max(df1['duration_min'].max(), df2['duration_min'].max()), 30)
        bin_labels = [format_min_minsec((bins[i] + bins[i+1])/2) for i in range(len(bins)-1)]

        # Group and aggregate track names for tooltips
        df1['bin'] = pd.cut(df1['duration_min'], bins=bins, labels=bin_labels, include_lowest=True)
        tooltip_data1 = df1.groupby('bin', observed=False)['track_name'].apply(list).reset_index()

        df2['bin'] = pd.cut(df2['duration_min'], bins=bins, labels=bin_labels, include_lowest=True)
        tooltip_data2 = df2.groupby('bin', observed=False)['track_name'].apply(list).reset_index()

        # Create subplots
        fig = make_subplots(rows=2, 
                            cols=1, 
                            shared_xaxes=True, 
                            subplot_titles=(label1, label2),
                            vertical_spacing=0.1)

        # Add histograms
        for idx, row in tooltip_data1.iterrows():
            fig.add_trace(go.Bar(
                x=[row['bin']], y=[len(row['track_name'])], 
                name=label1, 
                hoverinfo='text', 
                text=["<br>".join(row['track_name'])],
                hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
                marker=dict(color=color_1),
                showlegend=False  
            ), row=1, col=1)

        for idx, row in tooltip_data2.iterrows():
            fig.add_trace(go.Bar(
                x=[row['bin']], y=[len(row['track_name'])], 
                name=label2, 
                hoverinfo='text', 
                text=["<br>".join(row['track_name'])],
                hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
                marker=dict(color=color_2),
                showlegend=False  
            ), row=2, col=1)

        def find_closest_bin(duration, bins, bin_labels):
            # Find the index of the closest bin
            index = np.digitize(duration, bins) - 1
            # Clamp index to valid range
            index = max(0, min(index, len(bin_labels) - 1))
            return bin_labels[index]
        
        # Mean duration line
        mean_duration1 = df1['duration_min'].mean()
        mean_duration2 = df2['duration_min'].mean()

        # Find the closest bin labels for the mean durations
        mean_bin_label1 = find_closest_bin(mean_duration1, bins, bin_labels)
        mean_bin_label2 = find_closest_bin(mean_duration2, bins, bin_labels)

        # Add Mean line
        fig.add_trace(go.Scatter(
            x=[mean_bin_label1, mean_bin_label1], 
            y=[0, tooltip_data1['track_name'].apply(len).max()],
            mode='lines', 
            name=f'Mean {label1}', 
            line=dict(color='red', dash='dash'),
            hoverinfo='text',
            hovertext=f"{mean_duration1:.2f}"
        ), row=1, col=1)

        # Add Mean line
        fig.add_trace(go.Scatter(
            x=[mean_bin_label2, mean_bin_label2], 
            y=[0, tooltip_data2['track_name'].apply(len).max()],
            mode='lines', 
            name=f'Mean {label2}', 
            line=dict(color='blue', dash='dash'),
            hoverinfo='text',
            hovertext=f"{mean_duration2:.2f}"
        ), row=2, col=1)

        # Update layout
        fig.update_layout(
            #xaxis_title='Duration (min:sec)',
            yaxis_title='Frequency',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            legend=dict(
                orientation='h',
                x=0.5,
                y=1.1  
            ),
            height=800,
            showlegend=True,
            autosize=True
        )

        return fig
        

    def tempo_histogram(self, df1, df2, label1, label2) -> go.Figure:
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
        color_1 = px.colors.sequential.Plasma[2]
        color_2 = px.colors.sequential.Plasma[6]

        # Define bins for histogram
        all_tempo = pd.concat([df1['tempo'], df2['tempo']])
        bins = np.linspace(all_tempo.min(), all_tempo.max(), 30)

        # Create bin labels formatted to two decimal places
        #bin_labels = [f"{(bins[i] + bins[i+1])/2:.2f}" for i in range(len(bins)-1)]

        # Convert tempo to bin
        df1['bin'] = pd.cut(df1['tempo'], bins=bins, include_lowest=True)
        df2['bin'] = pd.cut(df2['tempo'], bins=bins, include_lowest=True)

        # Group and aggregate track names for tooltips
        tooltip_data1 = df1.groupby('bin', observed=False)['track_name'].apply(list).reset_index()
        tooltip_data2 = df2.groupby('bin', observed=False)['track_name'].apply(list).reset_index()

        # Format bin labels for x-axis
        #tooltip_data1['bin'] = tooltip_data1['bin'].apply(lambda x: f"{x.left:.2f} - {x.right:.2f}")
        #tooltip_data2['bin'] = tooltip_data2['bin'].apply(lambda x: f"{x.left:.2f} - {x.right:.2f}")
        
        # Initialize subplots
        fig = make_subplots(rows=2,
                            cols=1,
                            shared_xaxes=True,
                            subplot_titles=(label1, label2),
                            vertical_spacing=0.1)
        
        # Add histogram
        for idx, row in tooltip_data1.iterrows():
            bin_label = f"{row['bin'].left:.2f} - {row['bin'].right:.2f}"  # Format bin label
            fig.add_trace(go.Bar(
                x=[bin_label], y=[len(row['track_name'])],
                name=label1,
                hoverinfo='text',
                text=["<br>".join(row['track_name'])],
                hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
                marker=dict(color=color_1),
                showlegend=False
            ), row=1, col=1)

        for idx, row in tooltip_data2.iterrows():
            bin_label = f"{row['bin'].left:.2f} - {row['bin'].right:.2f}"  # Format bin label
            fig.add_trace(go.Bar(
                x=[bin_label], y=[len(row['track_name'])],
                name=label2,
                hoverinfo='text',
                text=["<br>".join(row['track_name'])],
                hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
                marker=dict(color=color_2),
                showlegend=False
            ), row=2, col=1)

        # Add mean lines
        mean1 = df1['tempo'].mean()
        mean2 = df2['tempo'].mean()

        # Find bins for the means
        mean1_bin = tooltip_data1.loc[tooltip_data1['bin'].apply(lambda x: x.left <= mean1 <= x.right), 'bin'].values[0]
        mean1_bin_label = f"{mean1_bin.left:.2f} - {mean1_bin.right:.2f}"
        mean2_bin = tooltip_data2.loc[tooltip_data2['bin'].apply(lambda x: x.left <= mean2 <= x.right), 'bin'].values[0]
        mean2_bin_label = f"{mean2_bin.left:.2f} - {mean2_bin.right:.2f}"

        fig.add_trace(go.Scatter(
            x=[mean1_bin_label, mean1_bin_label],
            y=[0, tooltip_data1['track_name'].apply(len).max()],
            mode='lines',
            name=f"Mean {label1}", 
            line=dict(color='red', dash='dash'),
            hoverinfo='text',
            hovertext=f"{mean1:.2f} BPM",
            showlegend=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[mean2_bin_label, mean2_bin_label], 
            y=[0, tooltip_data2['track_name'].apply(len).max()],
            mode='lines', 
            name=f'Mean {label2}', 
            line=dict(color='blue', dash='dash'),
            hoverinfo='text',
            hovertext=f"{mean2:.2f} BPM",
            showlegend=True
        ), row=2, col=1)

        # Update layout
        fig.update_layout(
            #xaxis_title='Tempo (BPM)',
            yaxis_title='Frequency',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            legend=dict(
                orientation='h',
                x=0.5,
                y=1.1
            ),
            height=800,
            showlegend=True,
            autosize=True
        )

        return fig

    
    def key_distribution(self, df1, df2, label1, label2) -> go.Figure:
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

        colorscale = [
            [0.0, "rgba(232, 148, 88, 0.8)"],
            [0.12, "rgba(213, 120, 98, 0.8)"],
            [0.24, "rgba(190, 97, 111, 0.8)"],
            [0.36, "rgba(164, 77, 126, 0.8)"],
            [0.48, "rgba(136, 60, 137, 0.8)"],
            [0.58, "rgba(125, 50, 140, 0.8)"],
            [0.68, "rgba(106, 44, 141, 0.8)"],
            [0.78, "rgba(87, 35, 142, 0.8)"],
            [0.88, "rgba(69, 27, 140, 0.8)"],
            [0.94, "rgba(40, 16, 137, 0.8)"],
            [0.97, "rgba(26, 12, 135, 0.8)"],
            [1.0, "rgba(12, 7, 134, 0.8)"]
        ]

        # Prepare data function
        def prepare_data(df):
            grouped_tracks = df.groupby('key')['track_name'].apply(list).reindex(range(12), fill_value=[])
            key_counts = df['key'].value_counts().reindex(range(12), fill_value=0)
            key_df = pd.DataFrame({
                'Key Name': [key_mapping[k] for k in range(12)],
                'Count': key_counts.values,
                'Track Names': ['<br>'.join(tracks) for tracks in grouped_tracks]
                })
            return key_df.sort_values(by='Count', ascending=False)
        
        key_df1 = prepare_data(df1)
        key_df2 = prepare_data(df2)

        # Create subplots
        fig = make_subplots(rows=1,
                            cols=2,
                            subplot_titles=(label1, label2),
                            shared_yaxes=True)
        
        fig.add_trace(go.Bar(
            x=key_df1['Key Name'],
            y=key_df1['Count'],
            name=label1,
            marker=dict(color=key_df1['Count'], colorscale=colorscale),
            hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
            text=key_df1['Track Names']
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=key_df2['Key Name'],
            y=key_df2['Count'],
            name=label2,
            marker=dict(color=key_df2['Count'], colorscale=colorscale),
            hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
            text=key_df2['Track Names']
        ), row=1, col=2)

        # Adding annotations for central x-axis title
        fig.add_annotation(
            x=0.5, y=-0.15,
            xref="paper", yref="paper",
            showarrow=False,
            text="Key",
            font=dict(size=14),
            align="center"
        )

        # Update layout
        fig.update_layout(
            showlegend=False,
            xaxis_title='', # remove xaxis title
            yaxis_title='Count',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            autosize=True,
            margin=dict(l=20, r=20, t=30, b=20)
        )

        return fig

    def loudness_histogram(self, df1, df2, label1, label2) -> go.Figure:
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
        color_1 = px.colors.sequential.Plasma[2]
        color_2 = px.colors.sequential.Plasma[6]

        # Define histogram
        all_loudness = pd.concat([df1['loudness'], df2['loudness']])
        bin_edges = np.linspace(all_loudness.min(), all_loudness.max(), 30)
        bin_labels = [f"{left:.2f} dB - {right:.2f} dB" for left, right in zip(bin_edges[:-1], bin_edges[1:])]


        # Group and aggregate track names for tooltips
        df1['bin'] = pd.cut(df1['loudness'], bins=bin_edges, include_lowest=True)
        df2['bin'] = pd.cut(df2['loudness'], bins=bin_edges, include_lowest=True)
        tooltip_data1 = df1.groupby('bin')['track_name'].apply(list).reset_index()
        tooltip_data2 = df2.groupby('bin')['track_name'].apply(list).reset_index()

        # Initialize subplots
        fig = make_subplots(rows=2,
                            cols=1,
                            shared_xaxes=True,
                            subplot_titles=(label1,label2),
                            vertical_spacing=0.1)

        # Add histogram
        for idx, row in tooltip_data1.iterrows():
            fig.add_trace(go.Bar(
                x=[bin_labels[idx]], 
                y=[len(row['track_name'])], 
                name=label1,
                hoverinfo='text', 
                text=["<br>".join(row['track_name'])],
                hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
                marker=dict(color=color_1), 
                showlegend=False),
                row=1, col=1)

        for idx, row in tooltip_data2.iterrows():
            fig.add_trace(go.Bar(
                x=[bin_labels[idx]], 
                y=[len(row['track_name'])], 
                name=label2,
                hoverinfo='text', 
                text=["<br>".join(row['track_name'])],
                hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
                marker=dict(color=color_2), 
                showlegend=False), 
                row=2, col=1)
        
        # Define means
        mean1 = df1['loudness'].mean()
        mean2 = df2['loudness'].mean()
        print("Mean1:", mean1, "Mean2:", mean2)

        # Function to find the closest bin label
        def find_closest_bin(duration, bins, bin_labels):
            index = np.digitize([duration], bins) - 1  # Use digitize to find appropriate bin
            index = max(0, min(index[0], len(bin_labels) - 1))  # Clamp index to valid range
            return bin_labels[index]

        mean1_label = find_closest_bin(mean1, bin_edges, bin_labels)
        mean2_label = find_closest_bin(mean2, bin_edges, bin_labels)

        max_y1 = tooltip_data1['track_name'].apply(len).max()
        max_y2 = tooltip_data2['track_name'].apply(len).max()

        # Add mean lines
        fig.add_trace(go.Scatter(
            x=[mean1_label, mean1_label], 
            y=[0, max_y1],
            mode='lines', 
            name=f"Mean {label1}", 
            line=dict(color='red', dash='dash'),
            hoverinfo='text', 
            hovertext=f"{mean1:.2f} dB", 
            showlegend=True), 
            row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=[mean2_label, mean2_label], 
            y=[0, max_y2],
            mode='lines', 
            name=f'Mean {label2}', 
            line=dict(color='blue', dash='dash'),
            hoverinfo='text', 
            hovertext=f"{mean2:.2f} dB", 
            showlegend=True), 
            row=2, col=1)

        # Update layout
        fig.update_layout(
            yaxis_title='Frequency',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            legend=dict(
                orientation='h',
                x=0.5,
                y=1.1
            ),
            height=800,
            showlegend=True,
            autosize=True
        )

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

        fig, axs = plt.subplots(2, 1, figsize=(14,10))
        axs[0].imshow(wc1, interpolation='bilinear')
        axs[0].axis("off")
        axs[0].set_title(label1)

        axs[1].imshow(wc2, interpolation='bilinear')
        axs[1].axis("off")
        axs[1].set_title(label2)
        
        plt.subplots_adjust(hspace=0.15)
        #plt.tight_layout(pad=0)
        return fig
    

    def mode_pie_chart(self, df1, df2, pl1, pl2) -> go.Figure:
        """ 
        Creates side-by-side pie charts comparing the mode (Major vs Minor) percentages 
        between two playlists.

        Args:
        df1 : pandas.DataFrame
        df2 : pandas.DataFrame
        pl1 : str
        pl2 : str

        Returns:
        plt.Figure
        """
        color_1 = px.colors.sequential.Plasma[2]
        color_2 = px.colors.sequential.Plasma[6]

        def prepare_data(df):
            df['mode'] = df['mode'].map({1: 'Major', 0: 'Minor'})
            title_summary = df.groupby('mode', observed=False)['track_name'].apply(lambda x: '<br>'.join(x[:5]) + ('...' if len(x) > 5 else '')).reset_index()
            title_summary.columns=['mode', 'titles']

            count_summary = df['mode'].value_counts().reset_index()
            count_summary.columns = ['mode', 'count']

            final_data = pd.merge(title_summary, count_summary, on='mode')
            #print(final_data)
            return final_data
        
        data1 = prepare_data(df1)
        data2 = prepare_data(df2)

        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]])
        colors = {'Major': color_1, 'Minor': color_2}

        # Plot for playlist 1
        fig.add_trace(
            go.Pie(labels=data1['mode'], 
                   values=data1['count'], 
                   name=pl1, 
                   marker_colors = [colors[label] for label in data1['mode']],
                   customdata=data1['titles'],
                   hovertemplate="<b>%{label}</b><br>Count: %{value}<br><b>Tracks:</b><br>%{customdata}<extra></extra>"),
            row=1, col=1
        )

        # Plot for playlist 2
        fig.add_trace(
            go.Pie(labels=data2['mode'], 
                   values=data2['count'], 
                   name=pl2, 
                   marker_colors=[colors[label] for label in data2['mode']],
                   customdata=data2['titles'],
                   hovertemplate="<b>%{label}</b><br>Count: %{value}<br><b>Tracks:</b><br>%{customdata}<extra></extra>"),
            row=1, col=2
        )

        # Update layout 
        fig.update_layout(
            # title_text=f"Mode Comparison: {pl1} vs {pl2}",
            annotations=[
            dict(text=pl1, x=0.18, y=1.1, font_size=12, showarrow=False, yanchor='bottom'),
            dict(text=pl2, x=0.82, y=1.1, font_size=12, showarrow=False, yanchor='bottom')],
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            #margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True,
            legend=dict(
                orientation='h',
                x=0.7,
                y=-0.1  
            ),
            autosize=True

        )

        return fig

    
    def explicit_pie_chart(self, df1, df2, pl1, pl2) -> go.Figure:
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
        color_1 = px.colors.sequential.Plasma[2]
        color_2 = px.colors.sequential.Plasma[6]

        def get_explicit_data(df):
            df['is_explicit'] = df['is_explicit'].map({True: 'Explicit', False: 'Non-Explicit'})
            title_summary = df.groupby('is_explicit')['track_name'].apply(lambda x: '<br>'.join(x[:5]) + ('...' if len(x) > 5 else '')).reset_index()
            title_summary.columns=['is_explicit', 'titles']

            count_summary = df['is_explicit'].value_counts().reset_index()
            count_summary.columns = ['is_explicit', 'count']

            final_data = pd.merge(title_summary, 
                                  count_summary, 
                                  on='is_explicit',
                                  how='right').fillna("No tracks available")
            print(final_data)
            return final_data
        
        data1 = get_explicit_data(df1)
        data2 = get_explicit_data(df2)
       
        labels = ['Explicit', 'Non-Explicit']
        colors = [color_1, color_2] 

        # Initialize subplots
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]])

        # Plot for playlist 1
        fig.add_trace(
            go.Pie(labels=data1['is_explicit'],
                values=data1['count'],
                name=pl1,
                marker_colors=[colors[i] for i in range(len(data1['is_explicit']))],
                textinfo='percent',
                hoverinfo='label+percent',
                customdata=data1['titles'],
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br><b>Tracks:</b><br>%{customdata}<extra></extra>"),
            row=1, col=1
        )
                
        # Pie chart for playlist 2
        fig.add_trace(
            go.Pie(labels=data2['is_explicit'],
                values=data2['count'],
                name=pl2,
                marker_colors=[colors[i] for i in range(len(data2['is_explicit']))],
                textinfo='percent',
                hoverinfo='label+percent',
                customdata=data2['titles'],
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br><b>Tracks:</b><br>%{customdata}<extra></extra>"),
            row=1, col=2
        )

        # Update layout 
        fig.update_layout(
            # title_text=f"Mode Comparison: {pl1} vs {pl2}",
            annotations=[
            dict(text=pl1, x=0.18, y=1.1, font_size=12, showarrow=False, yanchor='bottom'),
            dict(text=pl2, x=0.82, y=1.1, font_size=12, showarrow=False, yanchor='bottom')],
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            #margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True,
            legend=dict(
                orientation='h',
                x=0.7,
                y=-0.1  
            ),
            autosize=True
        )

        return fig


    def run_analysis(self, id1, id2):
        att1, df1, att2, df2 = self.compare_playlists(id1, id2)
        label1 = self.get_playlist_name(id1)  # Placeholder function
        label2 = self.get_playlist_name(id2) 

        st.header('Era DataFrames')
        st.text(f"{label1} DataFrame")
        st.dataframe(df1)
        st.text(f"{label2} DataFrame")
        st.dataframe(df2)
        
        st.header('Radar Chart Comparison:')
        st.text("Music Era Comparison of Attributes (Mean Values)")
        radarchart = self.radar_chart(att1, att2, label1, label2)
        st.plotly_chart(radarchart, use_container_width=True)

        st.header('Duration Histogram Comparison:')
        st.text("Music Era Comparison of Track Duration")
        durationhist = self.duration_histogram(df1, df2, label1, label2)
        st.plotly_chart(durationhist, use_container_width=True)

        st.header('Tempo (BPM) Histogram Comparision:')
        st.text("Music Era Comparision of Tempo")
        tempohist = self.tempo_histogram(df1, df2, label1, label2)
        st.plotly_chart(tempohist, use_container_width=True)

        st.header('Key Distribution Comparision:')
        st.text("Music Era Comparision of Key")
        keybar = self.key_distribution(df1, df2, label1, label2)
        st.plotly_chart(keybar, use_container_width=True)

        st.header('Loudness (dB) Histogram Comparison:')
        st.text("Music Era Comparision of Loudness")
        loudhist = self.loudness_histogram(df1, df2, label1, label2)
        st.plotly_chart(loudhist, use_container_width=True)

        st.header('Artist Genres Word Cloud Comparison:')
        st.text("Music Era Comparison of Word Cloud")
        wcfig = self.genre_wordcloud(df1, df2, label1, label2)
        if wcfig is not None:
            st.pyplot(wcfig)
        else:
            st.error("Word Cloud could not be generated due to the insufficient data.")

        st.header('Mode Pie Chart Comparison:')
        st.text("Music Era Comparison of Mode (Major vs Minor)")
        mode_chart = self.mode_pie_chart(df1, df2, label1, label2)
        st.plotly_chart(mode_chart, use_container_width=True)

        st.header('Explicitness Pie Chart Comparision:')
        st.text("Music Era Comparision of Explicitness")
        explicit = self.explicit_pie_chart(df1, df2, label1, label2)
        st.plotly_chart(explicit, use_container_width=True)
        

# Initialize the Spotify client
sp = init_spotify_client()

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
st.info("You are comparing different music eras to explore how musical attributes and popular trends have evolved over time.", icon="👩🏽‍🎤")

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
