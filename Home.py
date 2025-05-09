import streamlit as st

st.set_page_config(
    page_title="Music Data Analysis App",
    page_icon="👋",
)

st.write("# Welcome to Music Data Analysis App 👋")

with st.sidebar:
    st.success("☝️Select an analysis from the sidebar to begin exploring music data.")
    st.subheader("🔥 **What's New?**")
    st.text("Version 0.5 Beta")
    st.markdown("""
    🌟 **Now Live:**
    - Artist Analysis
    - Song-To-Song Comparison
                
    🚧 **In Development:**
    - Genre Analysis
    - Playlist Analysis
    - Soothing Music Chatbot
        """)

st.markdown(
    """
    Music Data Analysis App, your go-to app for exploring and analyzing music data 
    Dive into the world of music through data-driven insights and interactive visualizations. 
    Whether you're a music enthusiast or a data analyst, this app offers a variety of analyses tailored to uncover the data stories behind the songs, artists, and trends shaping the music world. 
    Enhance your experience by listening to your chosen tracks or albums directly within the app, as you explore the rich data each piece of music holds.

    ## Features
    - **Top Chart Analysis:** Dive into the most popular tracks and artists dominating the charts.
    - **Track Recommendation:** Discover new music based on your favorite track.
    - **Popularity Check:** Analyze the popularity trends of songs and artists over time.
    - **Track Comparison:** Compare the attributes of different tracks to see what makes a hit.
    - **Era Comparison:** Explore how musical trends and preferences have evolved across different eras.
    - **Album Analysis:** Examine entire albums to understand the collective mood and technical attributes. 
    - **Fine-Tuned Recommendation:** Harness advanced controls to tailer music recommendations to your specific preferences. 
    - **Artist Analysis:** Gain deep insights into an artist's discography, popularity, and musical style.
    - **Song-to-Song Comparison:** Compare detailed attributes and popularity between two songs to uncover unique musical elements and trends.
    
    ## How to Use
    Ready to uncover the secrets of the music industry through data?  
    **👈 Select an analysis from the sidebar** and let the journey begin!

    ### Learn more about music data analysis app
    Whether you're new to the field or looking to expand your knowledge, check out these resources:
    - Delve into the [README doc](https://github.com/yuka-with-data/Spotify-Data-App-Public) for detailed guides on using the app.
    - Join the conversation in our community forum (Coming Soon)

    ### Inspired to create your own data project?
    Streamlit is an open-source app framework perfect for building interactive and shareable web apps for data science and machine learning projects.
    - Get started with Streamlit at [streamlit.io](https://streamlit.io)
    - Jump into Streamlit [documentation](https://docs.streamlit.io)

    #### Privacy Statement
    No personal data is logged by this app. No log-in required.
"""
)

st.divider()
st.caption("🛑Disclaimer: This application is designed for educational and entertainment purposes only. The data used in this app is sourced from public APIs and may not be entirely accurate or up-to-date. Embedded Spotify players are used under the terms of Spotify's embedding policies to enhance user experience by providing direct access to playlists and tracks. The analysis and visualizations provided should not be considered professional advice, and the creators of this app are not responsible for any decisions made based on the information provided. Please note that this app is not affiliated with Spotify or any other third-party services. Use this app responsibly and be aware of the terms and conditions of the Spotify API and the implications of embedded content. Enjoy exploring the data responsibly!")