# Load Libraries
import pandas as pd
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
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
st.set_page_config(page_title="Artist Analysis", 
                   page_icon="✨")

# Import data_galaxy after Page Config
from data_galaxy import init_spotify_client, fetch_artist_tracks

class SpotifyAnalyzer:
    def __init__(self, sp, artist_id: str) -> None:
        self.sp = sp
        self.mean_values_artist, self.df_artist = fetch_artist_tracks(self.sp, artist_id)
        self.colorscale=[  # Custom colorscale
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


    def radar_chart(self) -> go.Figure:
        color_artist = "rgba(69, 27, 140, 0.9)" # Rich Purple
        mean_values_artist = self.mean_values_artist * 100
        att_list = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
        fig = go.Figure()

        # Add trace (mean)
        fig.add_trace(go.Scatterpolar(
            r=mean_values_artist,
            theta=att_list,
            fill='toself',
            name='Chart (mean)',
            fillcolor=color_artist,
            line=dict(color=color_artist),
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
    
    def attribute_time_series(self) -> go.Figure:
        def compute_yearly_mean(df: pd.DataFrame) -> pd.DataFrame:
            # Handle invalid dates
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df = df.dropna(subset=['release_date'])
            
            df['year'] = pd.to_datetime(df['release_date']).dt.year
            yearly_means = df.groupby('year')[['energy', 'danceability', 'acousticness', 'valence', 'instrumentalness', 'liveness', 'speechiness']].mean().reset_index()
            return yearly_means.interpolate(method='linear')
        yearly_means = compute_yearly_mean(self.df_artist)

        fig = go.Figure()
        attributes = ['energy', 'danceability', 'acousticness', 'valence', 'instrumentalness', 'liveness', 'speechiness']
        colors = ["rgba(232, 148, 88, 0.9)", "rgba(213, 120, 98, 0.9)", "rgba(190, 97, 111, 0.9)", "rgba(164, 77, 126, 0.9)",
                  "rgba(136, 60, 137, 0.9)", "rgba(125, 50, 140, 0.9)", "rgba(106, 44, 141, 0.9)"]

        for att, color in zip(attributes, colors):
            fig.add_trace(go.Scatter(
                x=yearly_means['year'],
                y=yearly_means[att],
                mode='lines+markers',
                name=att,
                hovertemplate=f"<b>Year</b>: %{{x}}<br><b>{att}</b>: %{{y:.2f}}<extra></extra>",
                line=dict(color=color)
            ))
        
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Attribute Mean',
            legend_title='Attribute',
            legend=dict(
                orientation='h',
                xanchor='center',
                yanchor='bottom',
                y=5,
                x=0.5
            ),
            paper_bgcolor='WhiteSmoke',
            font={"color": "black"},
            height=450,
            width=700,
            margin=dict(l=40, r=40, t=40, b=40),
            autosize=True
        )

        return fig
    
    def attribute_heatmap(self) -> go.Figure:
        # Create correlation matrix df
        df_corr = self.df_artist[['energy', 'danceability', 'acousticness', 'valence', 'instrumentalness', 'liveness', 'speechiness']].corr()
        
        fig = px.imshow(
            df_corr,
            labels=dict(x="Features", y="Features", color="Correlation"),
            x=df_corr.columns,
            y=df_corr.index,
            color_continuous_scale='Spectral'
        )

        rounded_corr = np.around(df_corr.values, decimals=2)
        annotations = []
        for i, row in enumerate(rounded_corr):
            for j, value in enumerate(row):
                annotations.append(dict(
                    x=j,
                    y=i,
                    text=str(value),
                    showarrow=False,
                    font=dict(color='white'),
                    ))
                

        fig.update_layout(
            height=700,
            width=700,
            autosize=True,
            annotations=annotations
        )

        return fig
        

    def key_distribution_chart(self) -> go.Figure:
        # Mapping of numeric key values to corresponding alphabetic keys
        key_mapping = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

        grouped = self.df_artist.groupby('key', observed=False)['track_name'].apply(list).reindex(range(12), fill_value=[])

        # Get key counts and ensure all keys are present
        key_counts = self.df_artist['key'].value_counts().reindex(range(12), fill_value=0)
        
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

        # Add the bar trace with custom tooltips
        fig.add_trace(go.Bar(
            x=key_df_sorted['Key Name'],
            y=key_df_sorted['Count'],
            text=key_df_sorted['Formatted Tracks'],
            hovertemplate='<br><b>Tracks:</b><br>%{text}<extra></extra>',
            marker=dict(color=key_df_sorted['Count'], colorscale=self.colorscale)
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
    

    def run_analysis(self) -> None:
        try:
            # Create a DataFrame
            st.header('Artist Tracks Table')
            st.dataframe(self.df_artist)

            # Display Artist Genres
            st.header('Artist Genre')
            st.text("This section displays the top genres associated with the artist's tracks.")
            # Extract and process genres
            def extract_genres(df):
                genres = set()
                for genre_list in df['genres']:
                    individual_genres = genre_list.split(',')
                    for genre in individual_genres:
                        genres.add(genre.strip())
                return genres
            artist_genres = extract_genres(self.df_artist)
            artist_genres_list = list(artist_genres)
            # Create HTML for genres as badges with the specified background color
            badges_html = "".join([f"<span style='background-color:rgba(26, 12, 135, 0.9); color:#ffffff; padding:5px; border-radius:5px; margin:2px; display:inline-block;'>{genre}</span>" for genre in artist_genres_list])
            # Display the genres using HTML
            st.write(f"<div style='display: flex; flex-wrap: wrap; gap: 10px;'>{badges_html}</div>", unsafe_allow_html=True)

            # Create a Radar Chart
            st.header('Attributes Radar Chart')
            st.text("The radar chart displays the distribution of various musical attributes for the selected tracks.")
            fig = self.radar_chart()
            st.plotly_chart(fig, use_container_width=True)

            # Create Attribute Time Series
            st.header('Attribute Time Series')
            st.text("This time series displays the trends of various musical attributes over the years.")
            fig = self.attribute_time_series()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Attribute Heatmap')
            st.text("This heatmap visualizes the correlation between different musical features in a Spotify dataset.")
            fig = self.attribute_heatmap()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Key Distribution Chart')
            st.text("This chart shows the distribution of artist tracks across different keys.")
            fig = self.key_distribution_chart()
            st.plotly_chart(fig, use_container_width=True)


        except Exception as e:
            print(e)
            st.error("An error occurred during analysis. Please try again.")

# Initialize Spotify Client
sp = init_spotify_client()


def artist_search_func(sp,query) -> List[str]:
    result = sp.search(q=query, type='artist', limit=5)
    artists = [artist['name'] for artist in result['artists']['items']]
    return artists

# Function to get artist ID from artist name
def get_artist_id(sp, artist_name) -> str:
    result = sp.search(q=artist_name, type='artist', limit=1)
    if result['artists']['items']:
        return result['artists']['items'][0]['id']
    else:
        return None

with st.sidebar:
    st.title("Enter Artist Name")

    #placeholder
    selected_artist = None

    # Artist input
    selected_artist = st_searchbox(
        label="Select Artist", 
        key="artist_input", 
        search_function=lambda query: artist_search_func(sp, query),
        placeholder="Search for an artist..."
    )

    # If an artist is selected, fetch and display the artist ID
    if selected_artist:
        artist_id = get_artist_id(sp, selected_artist)
        st.write(f"Selected Artist: {selected_artist}")
        if artist_id:
            st.write(f"Artist ID: {artist_id}")
        else:
            st.write("Artist ID not found")

analyze_button = st.sidebar.button("Analyze")


#-------- Main ----------
st.markdown("# Artist Analysis")
st.info("Select an artist name to analyze. You'll get a comprehensive analysis of your selected album.", icon="📀")


if analyze_button:
    try:
        if artist_id:
            spotify_analyzer = SpotifyAnalyzer(sp, artist_id)
            artist_embed_url = f"https://open.spotify.com/embed/artist/{artist_id}?utm_source=generator&theme=0"
            st.components.v1.iframe(artist_embed_url, width=500, height=600, scrolling=True)
            st.divider()

            # Run Analysis
            spotify_analyzer.run_analysis()
            st.balloons()
        
        else:
            st.error("Cannot analyze. Artist ID is not available.")

    except Exception as e:
        print(e)
        st.error(f'An error occurred: {str(e)}')

elif analyze_button and not selected_artist:
    st.warning("Please choose a target artist to analyze.", icon='⚠️')