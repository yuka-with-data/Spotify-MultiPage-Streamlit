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
                marker=dict(color=color_1),
                showlegend=False  
            ), row=1, col=1)

        for idx, row in tooltip_data2.iterrows():
            fig.add_trace(go.Bar(
                x=[row['bin']], y=[len(row['track_name'])], 
                name=label2, 
                hoverinfo='text', 
                text=["<br>".join(row['track_name'])],
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
            x=[mean_bin_label1, mean_bin_label1], y=[0, tooltip_data1['track_name'].apply(len).max()],
            mode='lines', name=f'Mean {label1}', line=dict(color='red', dash='dash'),
        ), row=1, col=1)

        # Add Mean line
        fig.add_trace(go.Scatter(
            x=[mean_bin_label2, mean_bin_label2], 
            y=[0, tooltip_data2['track_name'].apply(len).max()],
            mode='lines', 
            name=f'Mean {label2}', 
            line=dict(color='blue', dash='dash'),
        ), row=2, col=1)

        # Update layout
        fig.update_layout(
            xaxis_title='Duration (min:sec)',
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
    

    def mode_pie_chart(self, df1, df2, pl1, pl2) -> plt.Figure:
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
        def get_mode_data(df):
            major_count = (df['mode'] == 1).sum() 
            minor_count = (df['mode'] == 0).sum()
            return [major_count, minor_count]
        
        # Get mode data from each playlist
        mode_1 = get_mode_data(df1)
        mode_2 = get_mode_data(df2)

        labels = ['Major', 'Minor']
        colors = [cm.plasma(0.10, alpha=0.65), cm.plasma(0.65, alpha=0.75)]

        # Initialize subplots for side-by-side pie charts
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].pie(mode_1,
                #labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12}, 
                radius=1.2)
        axs[0].set_title(pl1)
        axs[0].axis('equal')
        
        wedges2, _, _ = axs[1].pie(mode_2,
                #labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12}, 
                radius=1.2)
        axs[1].set_title(pl2)
        axs[1].axis('equal')

        fig.legend(wedges2, 
                   labels, 
                   title="Mode", 
                   loc="center right", 
                   fontsize='small'
                   )
        fig.patch.set_facecolor('lightgrey')
        plt.tight_layout()

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
        colors = [cm.plasma(0.10, alpha=0.65), cm.plasma(0.65, alpha=0.75)]

        # Initialize subplots for side-by-side pie charts
        fig, axs = plt.subplots(1,2,figsize=(12,6))

        axs[0].pie(explicit_1,
                   # labels=labels,
                   colors=colors,
                   autopct='%1.1f%%',
                   startangle=90,
                   textprops={'fontsize': 12}, 
                   radius=1.2)
        axs[0].set_title(pl1)
        axs[0].axis('equal')
        
        wedges2, _, _ = axs[1].pie(explicit_2,
                   # labels=labels,
                   colors=colors,
                   autopct='%1.1f%%',
                   startangle=90,
                   textprops={'fontsize': 12}, 
                   radius=1.2)
        axs[1].set_title(pl2)
        axs[1].axis('equal')

        fig.legend(wedges2, 
                   labels, 
                   title="Mode", 
                   loc="center right", 
                   fontsize='small'
                   )
        fig.patch.set_facecolor('lightgrey')
        plt.tight_layout()

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

        st.header('Mode Pie Chart Comparison:')
        st.text("Music Era Comparison of Mode (Major vs Minor)")
        mode_chart = self.mode_pie_chart(df1, df2, label1, label2)
        st.pyplot(mode_chart)

        st.header('Explicitness Pie Chart Comparision:')
        st.text("Music Era Comparision of Explicitness")
        explicit = self.explicit_pie_chart(df1, df2, label1, label2)
        st.pyplot(explicit)
        

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
