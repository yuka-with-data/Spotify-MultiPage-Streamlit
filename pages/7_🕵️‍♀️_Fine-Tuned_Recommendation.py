import streamlit as st
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from typing import List

# Page Config
st.set_page_config(page_title="Fine-Tuned Recommendation", 
                   page_icon="🔬")

# Import data_galaxy after Page Config
from data_galaxy import init_spotify_client

# Initialize Spotify Client
sp = init_spotify_client()

# Fetch available genres
def get_genres(sp) -> List[str]:
    try:
        return sp.recommendation_genre_seeds()['genres']
    except Exception as e:
        st.error(f"Error fetching genres: {e}")
        return []
genres = get_genres(sp)
print(genres)

# Define the key mapping dictionary
key_mapping = {
    0: "C",
    1: "C# / Db",
    2: "D",
    3: "D# / Eb",
    4: "E",
    5: "F",
    6: "F# / Gb",
    7: "G",
    8: "G# / Ab",
    9: "A",
    10: "A# / Bb",
    11: "B"
}

with st.sidebar:
    st.title("Customize Your Track Preferences")
    with st.form(key='my_form'):
        selected_genres = st.multiselect("Select Genres", options=genres, default=None)
        danceability = st.select_slider("Danceability", options=[i/100.0 for i in range(101)], value=0.50)
        energy = st.select_slider("Energy", options=[i/100.0 for i in range(101)], value=0.50)
        valence = st.select_slider("Valence (Musical Positivity)", options=[i/100.0 for i in range(101)], value=0.50)
        acousticness = st.select_slider("Acousticness", options=[i/100.0 for i in range(101)], value=0.50)
        instrumentalness = st.select_slider("Instrumentalness", options=[i/100.0 for i in range(101)], value=0.50)
        liveness = st.select_slider("Liveness", options=[i/100.0 for i in range(101)], value=0.50)
        speechiness = st.select_slider("Speechiness", options=[i/100.0 for i in range(101)], value=0.50)
        tempo = st.select_slider("Tempo (BPM)", options=range(50, 201), value=120)
        loudness = st.select_slider("Loudness (dB)", options=range(-60, 1), value=-30)
        key = st.select_slider("Key", options=list(key_mapping.values()), value="C")
        mode = st.selectbox("Mode", options=["Choose a Mode", "Major", "Minor"], index=0)
        exclude_explicit = st.checkbox("Exclude Explicit Tracks", value=False)
        submit_button = st.form_submit_button("Get Recommendations")

st.markdown("# Fine-Tuned Recommendation Tracks")
st.info("Select your target values and click the 'Get Recommendations' button to get personalized recommendations.", icon="🔬")

if submit_button:
    if not selected_genres:
        st.error("Please select at least one genre.")
    else:
        try:
            # Find the numerical key for the given note name (reverse mapping)
            reverse_key_mapping = {v: k for k, v in key_mapping.items()}
            numeric_key = reverse_key_mapping[key]
            mode_value = 1 if mode == "Major" else 0

            seeds = {'target_danceability': danceability,
                    'target_energy': energy,
                    'target_valence': valence,
                    'target_acousticness': acousticness,
                    'target_instrumentalness': instrumentalness,
                    'target_liveness': liveness,
                    'target_speechiness': speechiness,
                    'target_tempo': tempo,
                    'target_loudness': loudness,
                    'target_key': numeric_key,
                    'target_mode': mode_value,
                    'exclude_explicit': exclude_explicit}
            
            if selected_genres:
                recommendations = sp.recommendations(seed_genres=selected_genres, limit=10, **seeds)
            else:
                recommendations = sp.recommendations(limit=10, **seeds)
            
            if recommendations['tracks']:
                for idx, track in enumerate(recommendations['tracks'], start=1):
                    track_info = f"{idx}. **{track['name']}** by **{', '.join([artist['name'] for artist in track['artists']])}** | Popularity: **{track['popularity']}**"
                    st.write(track_info)
                    st.components.v1.iframe(f"https://open.spotify.com/embed/track/{track['id']}?utm_source=generator", width=500, height=160, scrolling=True)
            else:
                st.write("No Recommendation found. Adjust the sliders and try again.")
        
        except Exception as e:
            st.error(f"Error fetching recommendations: {str(e)}")

st.divider()
st.caption("""
**🛑Disclaimer on Recommendations:** 
The recommendations provided here are generated based on target values and other parameters you set. These values guide the search but do not guarantee that every track will exactly match the specified attributes. The range and flexibility in the recommendations ensure a variety of music that generally aligns with preferences you set but might slightly differ. Please note, these recommendations are generated algorithmically and should not be considered professional or definitive musical advice.
""")
            
