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
                   page_icon="💿")

# Import data_galaxy after Page Config
from data_galaxy import init_spotify_client, retrieve_playlist_data

class SpotifyAnalyzer:
    def __init__(self, sp, playlist_id: str) -> None:
        self.sp = sp
        self.mean_values_top_50, self.df_top_50 = retrieve_playlist_data(self.sp, playlist_id)

    def artist_bubble(self) -> go.Figure:
        artist_counts = self.df_top_50['artist_name'].value_counts().reset_index()
        artist_counts.columns = ['artist_name', 'frequency']

        # Average Popularity
        artist_popularity = self.df_top_50.groupby('artist_name', observed=False)['popularity'].sum().reset_index()
        # Merge the datasets
        merged_df = artist_counts.merge(artist_popularity, on='artist_name')

        total_popularity_max = merged_df['popularity'].max()
        sizeref_value = 2. * total_popularity_max / (100.**2)

        # Initialize visible text column
        merged_df['visible_text'] = ''
        # Set visible text only for top 3 frequent artists
        merged_df.loc[0, 'visible_text'] = merged_df.loc[0, 'artist_name']

        colorscale=[  # Custom colorscale
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

        # Create bubble chart
        fig = go.Figure(data=[go.Scatter(
            x=merged_df['frequency'],
            y=merged_df['popularity'],
            text=merged_df['visible_text'],  # conditionally display text
            hovertext=merged_df['artist_name'],  # display artist names on hover
            mode='markers+text',  # show markers and text
            # textposition='top center',
            marker=dict(
                size=merged_df['popularity'],
                sizemode='area',
                sizeref=sizeref_value,
                sizemin=4,
                color=merged_df['popularity'],
                colorbar=dict(title='Score', thickness=10),
                colorscale=colorscale 
            ),
            hoverinfo='text+x+y',
            hovertemplate='<b>%{hovertext}</b><br>Frequency: %{x}<br>Popularity: %{y}<extra></extra>'
        )])

        fig.update_layout(
            # title='Artist Presence and Popularity in the Top Chart',
            xaxis_title='Frequency of Artist Appearance',
            yaxis_title='Popularity Score',
            font=dict(color="black"),  # Ensuring all chart text is black
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            height=600,
            width=700,
            paper_bgcolor = 'Gainsboro',
            plot_bgcolor='Gainsboro',
            autosize=True
        )

        return fig
    
    def artist_sankey(self) -> go.Figure:

        def prepare_sankey_data():
            # Aggregate popularity scores by artist
            artist_popularity = self.df_top_50.groupby('artist_name', observed=False)['popularity'].sum().reset_index()
            artist_popularity['popularity_category'] = pd.qcut(artist_popularity['popularity'], 3, labels=["Low", "Medium", "High"])
            #print(artist_popularity)
            return artist_popularity
        
        artist_popularity = prepare_sankey_data()

        # Create lists of unique nodes
        artists = artist_popularity['artist_name'].tolist()
        categories = ['High', 'Medium', 'Low']

        # Map to indices for Sankey diagram
        artist_indices = {artist: i for i, artist in enumerate(artists)}
        category_indices = {category: i + len(artists) for i, category in enumerate(categories)}
        #print(category_indices)

        # Define nodes
        nodes = [{'label': artist} for artist in artists] + [{'label': category} for category in categories]

        # Define links
        links = [{
            'source': artist_indices[row['artist_name']],
            'target': category_indices[row['popularity_category']],
            'value': row['popularity']
        } for index, row in artist_popularity.iterrows()]

        # Create the Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[node['label'] for node in nodes],
                color='rgba(69, 27, 140, 0.9)', #purple
                hoverinfo="all",
                hovertemplate='<b>%{label}</b>',
                #font=dict(color='white', size=10) 
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links],
                color='rgba(255, 127, 80, 0.7)', # orange
                hoverinfo="skip"
            )
        )])

        fig.update_layout(# title_text="Sankey Diagram:",
                          height=600,
                          width=700,
                        font_size=10,
                        margin=dict(l=20, r=20, t=20, b=20),
                        autosize=True,
                        paper_bgcolor='Gainsboro',
                        
                        )
        return fig

    
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
            name='Chart (mean)',
            fillcolor=color_top_50,
            line=dict(color=color_top_50),
        ))

        # Update the layout 
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            legend=dict(x=1, y=1, traceorder='normal',orientation='v', font=dict(color='black')),
            paper_bgcolor = 'Gainsboro',
            font = {"color": "black"},
            height=450,
            width=700,
            autosize=True
        )

        return fig
        
    def tempo_histogram(self) -> go.Figure:
        color_top_50 = px.colors.sequential.Plasma[2]  
        color_average_tempo = px.colors.sequential.Plasma[6]  

        # Calculate histogram data manually to determine the max frequency
        data_min = self.df_top_50['tempo'].min()
        data_max = self.df_top_50['tempo'].max()
        bins = np.linspace(data_min - 0.1, data_max + 0.1, num=50)
        counts, bins = np.histogram(self.df_top_50['tempo'], bins=bins)
        bins = np.round(bins, 2)  # Round bins to two decimal places

        # Group data into bins
        bin_labels = [f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)-1)]
        self.df_top_50['bin'] = pd.cut(self.df_top_50['tempo'], bins=bins, labels=bin_labels, include_lowest=True)

        # Prepare data for the tooltips
        grouped = self.df_top_50.groupby('bin', observed=False)
        tooltip_data = grouped['track_name'].agg(lambda x: ', '.join(x)).reset_index()

        # Create the figure and add bars manually
        fig = go.Figure()
        for label, group in grouped:
            fig.add_trace(go.Bar(
                x=[label], 
                y=[group['tempo'].count()],
                text=[tooltip_data[tooltip_data['bin'] == label]['track_name'].values[0]],
                hoverinfo="text",
                marker=dict(color=color_top_50, line=dict(width=1, color='black')),
                name=label,
                showlegend=False  # Hide legend for bars
            ))

        # Calculate mean tempo
        mean_tempo = self.df_top_50['tempo'].mean()
        # Find the bin label for the mean tempo
        mean_tempo_bin = pd.cut([mean_tempo], bins=bins, labels=bin_labels, include_lowest=True)[0]

        # Add a line for the average tempo
        fig.add_trace(go.Scatter(
            x=[mean_tempo_bin, mean_tempo_bin],
            y=[0, counts.max()],  # Use the maximum count as the top of the line
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
            plot_bgcolor='Gainsboro',
            paper_bgcolor='Gainsboro',
            # legend_title_text='Legend',
            legend=dict(
                orientation='h',
                y=1.1
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            autosize=True
        )

        return fig
    
    def duration_histogram(self) -> plt.Figure:

        color_top_50 = px.colors.sequential.Plasma[2]  
        color_average_duration = px.colors.sequential.Plasma[6]  

        # Convert duration from milliseconds to minutes for readability
        self.df_top_50['duration_min'] = self.df_top_50['duration_ms'] / 60000

        # Calculate histogram data
        data_min = self.df_top_50['duration_min'].min()
        data_max = self.df_top_50['duration_min'].max()
        bins = np.linspace(data_min - 0.1, data_max + 0.1, num=50)
        counts, bins = np.histogram(self.df_top_50['duration_min'], bins=bins)
        bins = np.round(bins, 2)  # Round bins to two decimal places

        # Create bin labels for grouping
        bin_labels = [f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)-1)]

        # Group by bins
        self.df_top_50['bin'] = pd.cut(self.df_top_50['duration_min'], bins=bins, labels=bin_labels, include_lowest=True)
        grouped = self.df_top_50.groupby('bin', observed=False)

        tooltip_data = grouped['track_name'].agg(lambda x: ', '.join(x)).reset_index()

        # Create the figure and add histogram bars manually
        fig = go.Figure()
        for label, group in grouped:
            fig.add_trace(go.Bar(
                x=[label], 
                y=[group['duration_min'].count()],
                text=[tooltip_data[tooltip_data['bin'] == label]['track_name'].values[0]],
                hoverinfo="text",
                marker=dict(color=color_top_50, line=dict(width=1, color='black')),
                name=label,
                showlegend=False  # Hide legend for bars
            ))

        # Calculate mean duration
        mean_duration = self.df_top_50['duration_min'].mean()
        # Find the bin label for the mean duration
        mean_duration_bin = pd.cut([mean_duration], bins=bins, labels=bin_labels, include_lowest=True)[0]

        # Add a line for the average duration
        fig.add_trace(go.Scatter(
            x=[mean_duration_bin, mean_duration_bin],
            y=[0, counts.max()],  # Use the maximum count as the top of the line
            mode='lines',
            line=dict(color=color_average_duration, width=2, dash='dash'),
            name='Average Duration',
            hoverinfo='text',
            text=f"Mean Duration: {mean_duration:.2f} min" 
        ))

        # Update layout with additional options
        fig.update_layout(
            xaxis_title='Duration (min)',
            yaxis_title='Frequency',
            template='plotly_white',
            plot_bgcolor='Gainsboro',
            paper_bgcolor='Gainsboro',
            legend=dict(
                orientation='h',
                y=1.1  # Adjust position of the legend
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            autosize=True
        )

        return fig

    
    def loudness_histogram(self) -> plt.Figure:
        color_top_50 = px.colors.sequential.Plasma[2]  
        color_average_loudness = px.colors.sequential.Plasma[6]  

        # Calculate histogram data
        data_min = self.df_top_50['loudness'].min()
        data_max = self.df_top_50['loudness'].max()
        bins = np.linspace(data_min - 0.1, data_max + 0.1, num=50)
        counts, bins = np.histogram(self.df_top_50['loudness'], bins=bins)
        bins = np.round(bins, 2)  # Round bins to two decimal places

        # Create bin labels for grouping
        bin_labels = [f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)-1)]
        self.df_top_50['bin'] = pd.cut(self.df_top_50['loudness'], bins=bins, labels=bin_labels, include_lowest=True)

        # Prepare data for the tooltips
        grouped = self.df_top_50.groupby('bin', observed=False)
        tooltip_data = grouped['track_name'].agg(lambda x: ', '.join(x)).reset_index()

        # Create the figure and add histogram bars manually
        fig = go.Figure()
        for label, group in grouped:
            fig.add_trace(go.Bar(
                x=[label], 
                y=[group['loudness'].count()],
                text=[tooltip_data[tooltip_data['bin'] == label]['track_name'].values[0]],
                hoverinfo="text",
                marker=dict(color=color_top_50, line=dict(width=1, color='black')),
                name=label,
                showlegend=False  # Hide legend for bars
            ))

        # Calculate mean loudness
        mean_loudness = self.df_top_50['loudness'].mean()
        # Find the bin label for the mean loudness
        mean_loudness_bin = pd.cut([mean_loudness], bins=bins, labels=bin_labels, include_lowest=True)[0]

        # Add a line for the average loudness
        fig.add_trace(go.Scatter(
            x=[mean_loudness_bin, mean_loudness_bin],
            y=[0, counts.max()],  # Use the maximum count as the top of the line
            mode='lines',
            line=dict(color=color_average_loudness, width=2, dash='dash'),
            name='Average Loudness',
            hoverinfo='text',
            text=f"Mean Loudness: {mean_loudness:.2f} dB" 
        ))

        # Update layout with additional options
        fig.update_layout(
            xaxis_title='Loudness (dB)',
            yaxis_title='Frequency',
            template='plotly_white',
            plot_bgcolor='Gainsboro',
            paper_bgcolor='Gainsboro',
            legend=dict(
                orientation='h',
                y=1.1  # Adjust position of the legend
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            autosize=True
        )

        return fig
    
    def key_distribution_chart(self) -> go.Figure:
        # Mapping of numeric key values to corresponding alphabetic keys
        key_mapping = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

        grouped = self.df_top_50.groupby('key', observed=False)['track_name'].apply(list).reindex(range(12), fill_value=[])

        # Get key counts and ensure all keys are present
        key_counts = self.df_top_50['key'].value_counts().reindex(range(12), fill_value=0)
        
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
            hovertemplate='%{text}<extra></extra>',
            marker=dict(color=key_df_sorted['Count'], colorscale=colorscale)
        ))

        # Update layout
        fig.update_layout(
            # title='Key Distribution',
            xaxis_title='Key',
            yaxis_title='Count',
            template='plotly_white',
            plot_bgcolor='Gainsboro',
            paper_bgcolor='Gainsboro',
            autosize=True,
            margin=dict(l=20, r=20, t=30, b=20)
        )

        return fig
    
    def mode_pie_chart(self):
        # Prepare data for Major vs. Minor mode
        mode_data = self.df_top_50[['mode', 'track_name']].copy()
        mode_data['mode'] = mode_data['mode'].map({1: 'Major', 0: 'Minor'})

        # Aggregate track names for each category
        title_summary = mode_data.groupby('mode', observed=False)['track_name'].apply(lambda x: '\n'.join(x[:5]) + ('...' if len(x) > 5 else '')).reset_index()
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
            plot_bgcolor='Gainsboro',
            paper_bgcolor='Gainsboro',
            autosize=True,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20)
        )

        return fig

    
    def explicit_pie_chart(self) -> plt.Figure:
        # Prepare data for Explicit vs. Non-Explicit
        explicit_data = self.df_top_50[['is_explicit', 'track_name']].copy()
        explicit_data['is_explicit'] = explicit_data['is_explicit'].map({True: 'Explicit', False: 'Non-Explicit'})

        # Aggregate track names for each category
        title_summary = explicit_data.groupby('is_explicit', observed=False)['track_name'].apply(lambda x: '\n'.join(x[:5]) + ('...' if len(x) > 5 else '')).reset_index()
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
            plot_bgcolor='Gainsboro',
            paper_bgcolor='Gainsboro',
            autosize=True,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20)
        )

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
            st.header('Top Chart Table')
            st.dataframe(self.df_top_50)

            st.header('Artist Presence in the Top Chart')
            st.text("This bubble chart shows the Artist Presence and Aggregated Popularity Score in the latest Top Chart")
            artist_bubble = self.artist_bubble()
            st.plotly_chart(artist_bubble, use_container_width=True)
            
            st.text("This Sankey Diagram shows the Artist Presence and Aggregated Popularity Score in the latest Top Chart")
            artist_sankey = self.artist_sankey()
            st.plotly_chart(artist_sankey, use_container_width=True)

            # Create a Radar Chart
            st.header('Attributes Radar Chart')
            st.text("The radar chart displays the distribution of various musical attributes for the selected tracks.")
            fig = self.radar_chart()
            st.plotly_chart(fig, use_container_width=True)

            # Create a Genre Word Cloud
            st.header('Genres Word Cloud')
            st.text("The word cloud illustrates the prevalence of various genres in the playlist based on text data.")
            wc = self.genres_wordcloud()
            st.pyplot(wc)

            # Create a BPM Histogram Chart
            st.header('Tempo Histogram Chart')
            st.text("This histogram shows the distribution of tempo (beats per minute) across tracks.")
            bpm_hist_chart = self.tempo_histogram()
            st.plotly_chart(bpm_hist_chart, use_container_width=True)

            st.header('Duration Histogram Chart')
            st.text("The histogram below represents the distribution of track durations in the playlist.")
            duration_dist = self.duration_histogram()
            st.plotly_chart(duration_dist, use_container_width=True)

            # Create a Key Distribution
            st.header('Key Distribution Comparison:')
            st.text("This chart compares the key distribution of the tracks, showing which musical keys are most common.")
            key_dist = self.key_distribution_chart()
            st.plotly_chart(key_dist, use_container_width=True)

            # Create a Duration Histogram Chart
            st.header('Loudness Histogram Chart')
            st.text("The loudness histogram visualizes the loudness levels of tracks in decibels.")
            loudness_hist_chart = self.loudness_histogram()
            st.plotly_chart(loudness_hist_chart, use_container_width=True)

            # Create a Mode Distribution
            st.header('Mode Pie Chart')
            st.text("Major modes are bright and uplifting, while minor modes are somber and serious.")
            mode_dist = self.mode_pie_chart()
            st.plotly_chart(mode_dist, use_container_width=True)

            # Create an Explicit Pie Chart
            st.header('Explicitness Pie Chart')
            st.text("This pie chart shows the proportion of explicit and non-explicit tracks in the playlist.")
            explicit_chart = self.explicit_pie_chart()
            st.plotly_chart(explicit_chart, use_container_width=True)

        except Exception as e:
            print(e)
            st.error("An error occurred during analysis. Please try again.")


# Initialize Spotify Client
sp = init_spotify_client()


# Hot Chart playlist options
playlists = {
    "Select a Playlist": None, # Placeholder value
    "Billboard Hot 100": "6UeSakyzhiEt4NB3UAd6NQ",
    "Top 50 Global (Daily)": "37i9dQZEVXbMDoHDwVN2tF",
    "Top Songs Global (Weekly)": "37i9dQZEVXbNG2KDcFcKOF",
    "Big On Ineternet": "37i9dQZF1DX5Vy6DFOcx00",
    "Viral 50 Global (Daily)": "37i9dQZEVXbLiRSasKsNU9"
    
}

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

if not selected_playlist:
    st.warning("Please select a playlist from the sidebar.", icon='⚠️')
elif compare_button:
    try:
        if playlist_id:
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