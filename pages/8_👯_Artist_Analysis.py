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
from data_retriever import init_spotify_client, fetch_artist_tracks, retrieve_playlist_data

class SpotifyAnalyzer:
    def __init__(self, sp, artist_id: str) -> None:
        self.sp = sp
        self.mean_values_artist, self.df_artist = fetch_artist_tracks(self.sp, artist_id)
        self.att_list = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
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
        fig = go.Figure()

        # Add trace (mean)
        fig.add_trace(go.Scatterpolar(
            r=mean_values_artist,
            theta=self.att_list,
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
            #legend_title='Attribute',
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


    def tempo_distribution(self) -> go.Figure:
        # Calculate mean tempo for each album
        album_means = self.df_artist.groupby('album_name')['tempo'].mean().reset_index()
        album_means_sorted = album_means.sort_values(by='tempo', ascending=True)

        # Create the bar plot
        fig = go.Figure(go.Bar(
            x=album_means_sorted['tempo'],
            y=album_means_sorted['album_name'],
            orientation='h',
            marker=dict(color='rgba(26, 12, 135, 0.9)'),
            hoverinfo='x+y',
        ))

        # Update layout
        fig.update_layout(
            xaxis_title='Tempo (Mean)',
            yaxis_title='Album Name',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            autosize=True,
            margin=dict(l=150, r=20, t=30, b=20)
        )

        # Update y-axis to fit long album names by truncating labels
        fig.update_yaxes(
            tickmode='array',
            tickvals=self.df_artist['album_name'].unique(),
            ticktext=[name if len(name) <= 20 else name[:17] + '...' for name in self.df_artist['album_name'].unique()],
            automargin=True
        )

        return fig
    

    def duration_distribution(self) -> go.Figure:
        # Convert duration from milliseconds to minutes
        self.df_artist['duration_minutes'] = self.df_artist['duration_ms'] / 60000
        
        # Calculate mean duration for each album
        album_means = self.df_artist.groupby('album_name')['duration_minutes'].mean().reset_index()
        album_means_sorted = album_means.sort_values(by='duration_minutes', ascending=True)

        # Create the bar plot
        fig = go.Figure(go.Bar(
            x=album_means_sorted['duration_minutes'],
            y=album_means_sorted['album_name'],
            orientation='h',
            marker=dict(color='rgba(69, 27, 140, 0.9)'),
            hoverinfo='x+y',
        ))

        # Update layout
        fig.update_layout(
            xaxis_title='Duration (Mean)',
            yaxis_title='Album Name',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            autosize=True,
            margin=dict(l=150, r=20, t=30, b=20)
        )

        # Update y-axis to fit long album names by truncating labels
        fig.update_yaxes(
            tickmode='array',
            tickvals=self.df_artist['album_name'].unique(),
            ticktext=[name if len(name) <= 20 else name[:17] + '...' for name in self.df_artist['album_name'].unique()],
            automargin=True
        )

        return fig
    
    def loudness_distribution(self) -> go.Figure:
        # Calculate mean loudness for each album
        album_means = self.df_artist.groupby('album_name')['loudness'].mean().reset_index()
        album_means_sorted = album_means.sort_values(by='loudness', ascending=True)

        # Create the bar plot
        fig = go.Figure(go.Bar(
            x=album_means_sorted['loudness'],
            y=album_means_sorted['album_name'],
            orientation='h',
            marker=dict(color='rgba(136, 60, 137, 0.9)'),
            hoverinfo='x+y',
        ))

        # Update layout
        fig.update_layout(
            xaxis_title='Loudness (Mean)',
            yaxis_title='Album Name',
            template='plotly_white',
            plot_bgcolor='WhiteSmoke',
            paper_bgcolor='WhiteSmoke',
            autosize=True,
            margin=dict(l=150, r=20, t=30, b=20)
        )

        # Update y-axis to fit long album names by truncating labels
        fig.update_yaxes(
            tickmode='array',
            tickvals=self.df_artist['album_name'].unique(),
            ticktext=[name if len(name) <= 20 else name[:17] + '...' for name in self.df_artist['album_name'].unique()],
            automargin=True
        )

        fig.add_vrect(
            x0=-20, x1=-10, 
            fillcolor="LightSkyBlue", opacity=0.2, 
            layer="below", line_width=0,
            annotation_text="Quiet", annotation_position="top"
        )
        fig.add_vrect(
            x0=-10, x1=-5, 
            fillcolor="LightGreen", opacity=0.2, 
            layer="below", line_width=0,
            annotation_text="Moderate", annotation_position="top"
        )
        fig.add_vrect(
            x0=-5, x1=0, 
            fillcolor="LightSalmon", opacity=0.2, 
            layer="below", line_width=0,
            annotation_text="Loud", annotation_position="top"
        )

        return fig
    
    def mode_pie_chart(self) -> go.Figure:
        # Prepare data for Major vs. Minor mode
        mode_data = self.df_artist[['mode', 'track_name']].copy()
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
            color='mode',  # Ensure the color is based on 'mode'
            color_discrete_map={
                'Major': "rgba(26, 12, 135, 0.8)",  # Purple
                'Minor': "rgba(213, 120, 98, 0.8)"  # Orange
            },
            hole=0.2,
            custom_data=['titles']
        )

        # Setting tooltip to display track titles, each on a new line
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<br><b>Tracks:</b><br>%{customdata}<extra></extra>"
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
    
    def explicitness_pie_chart(self) -> go.Figure:
        # Prepare data for explicit vs. non-explicit tracks
        explicit_data = self.df_artist[['is_explicit', 'track_name']].copy()
        explicit_data['is_explicit'] = explicit_data['is_explicit'].map({True: 'Explicit', False: 'Non-Explicit'})

        # Aggregate track names for each category
        title_summary = explicit_data.groupby('is_explicit', observed=False)['track_name'].apply(lambda x: '<br>'.join(x[:5]) + ('...' if len(x) > 5 else '')).reset_index()
        title_summary.columns = ['is_explicit', 'titles']  # Correctly rename columns

        # Calculate counts for each category
        count_summary = explicit_data['is_explicit'].value_counts().reset_index()
        count_summary.columns = ['is_explicit', 'count']  # Directly assigning new column names

        # Merge titles and counts data
        final_data = pd.merge(title_summary, count_summary, on='is_explicit')

        # Create the pie chart
        fig = px.pie(
            final_data,
            names='is_explicit',
            values='count',
            #color_discrete_sequence=["rgba(26, 12, 135, 0.8)", "rgba(213, 120, 98, 0.8)"],
            color='is_explicit',# Ensure the color is based on 'is_explicit'
            color_discrete_map={
                'Explicit': "rgba(26, 12, 135, 0.8)",  # Purple
                'Non-Explicit': "rgba(213, 120, 98, 0.8)"  # Orange
            },
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
    

    def artist_popularity_gauge_chart(self) -> go.Figure:
        # Calculate popularity score mean
        def calculate_popularity():
            return self.df_artist['popularity'].mean()
        
        # Calculate the highest popularity score
        def calculate_highest_popularity():
            return self.df_artist['popularity'].max()
        
        #mean_popularity = calculate_popularity()
        highest_popularity = calculate_highest_popularity()

        # Define the categorize_popularity function
        def categorize_popularity(popularity_score):
            if popularity_score >= 90:
                return "Very Popular!"
            elif 70 <= popularity_score < 90:
                return "Popular"
            elif 50 <= popularity_score < 70:
                return "Moderately Popular"
            elif 30 <= popularity_score < 50:
                return "Not Very Popular"
            else:
                return "Not Popular"
        
        popularity_category = categorize_popularity(highest_popularity)

        # Create a gauge chart
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=highest_popularity,
            domain={'x': [0, 1], 'y': [0, 1]},
            #title={'text': "Artist Popularity Score"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth':1, 'tickcolor': "darkblue"},
                'bar': {'color': 'rgba(89, 42, 154, 1)'},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 100], 'color': 'GhostWhite'},
                    {'range': [0, highest_popularity], 'color': 'rgba(26, 12, 135, 0.5)'}
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

        # Add a text annotation for the popularity category
        fig.add_annotation(
            text=popularity_category,
            x=0.5,
            y=0.45,
            showarrow=False,
            font=dict(size=20, color="black"),
            align="center"
        )

        return fig
    
    def display_top_tracks(self):
        # Reorder the track by popularity score
        sorted_tracks = self.df_artist.sort_values(by='popularity', ascending=False)
        
        # Remove dupes
        unique_tracks = sorted_tracks.drop_duplicates(subset='track_name', keep='first')
        
        # Select top 10
        top_tracks = unique_tracks[['track_name', 'popularity']].head(10)

        # Add ranking index 
        top_tracks.index = range(1, len(top_tracks) + 1)
        top_tracks.index.name = 'Rank'
        
        return st.table(top_tracks)


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
                    if genre_list and genre_list.strip(): # Check if genre_list is not None, not empty, and not just whitespace
                        individual_genres = genre_list.split(',')
                        for genre in individual_genres:
                            genres.add(genre.strip())
                    return genres
            artist_genres = extract_genres(self.df_artist)
            artist_genres_list = list(artist_genres)

            if not artist_genres_list:
                st.warning("No genres found for this artist.")
            else:
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

            st.header('Tempo Bar Chart')
            st.text("This bar plot provides insights into the average tempo by albums.")
            fig = self.tempo_distribution()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Duration Bar Chart')
            st.text("This bar chart displays the average duration (in minutes) by albums.")
            fig = self.duration_distribution()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Loudness Bar Chart')
            st.text("This bar chart shows the average loudness by albums.")
            fig = self.loudness_distribution()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Mode Pie Chart')
            st.text("This pie chart illustrates the distribution of major and minor modes across the artist's tracks.")
            fig = self.mode_pie_chart()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Explicitness Pie Chart')
            st.text("This pie chart shows the distribution of explicit and non-explicit tracks across the artist's tracks.")
            fig = self.explicitness_pie_chart()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Artist Popularity Gauge Chart')
            st.text("This gauge chart displays today's popularity score (the average of all track's popularity scores) of the artist.")
            fig = self.artist_popularity_gauge_chart()
            st.plotly_chart(fig, use_container_width=True)

            st.header('Artist Top Tracks')
            st.text("This table shows the top 10 tracks of the artist.")
            self.display_top_tracks()

            st.subheader('Artist Presence in Billboard Top 100')
            st.text("This chart shows the presence of the artist in the Billboard Top 100 charts.")
            # set the ID
            billboard_playlist_id = "6UeSakyzhiEt4NB3UAd6NQ"
            _, df_top_100 = retrieve_playlist_data(self.sp, billboard_playlist_id)
            print(df_top_100)
            # Filter the tracks from the specified artist
            artist_in_100 = df_top_100[df_top_100['artist_name'] == selected_artist]

            if not artist_in_100.empty:
                # Highlight tracks by the specified artist
                def highlight_artist_tracks(row):
                    if row['artist_name'] == selected_artist:
                        return ['background-color: rgba(26, 12, 135, 0.5)'] * len(row)
                    else:
                        return [''] * len(row)
                
                artist_in_100_revised = artist_in_100.style.apply(highlight_artist_tracks, axis=1)
                st.dataframe(artist_in_100_revised)

            else:
                st.warning("Artist not found in the latest Billboard Top 100.")


        except Exception as e:
            print(e)
            st.error("An error occurred during analysis. Please try again.")

# Initialize Spotify Client
sp = init_spotify_client()


def artist_search_func(sp,query) -> List[str]:
    if not query:
        return []
    result = sp.search(q=query, type='artist', limit=5)
    artists = [artist['name'] for artist in result['artists']['items']]
    return artists

# Function to get artist ID from artist name
def get_artist_id(sp, artist_name) -> str:
    if not artist_name:
        return None
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
st.warning("It may take a long time to load the data for artists with 100+ tracks.")

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