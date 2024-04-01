import streamlit as st

st.set_page_config(
    page_title="Music dataGalax",
    page_icon="👋",
)

st.write("# Welcome to Music dataGalax! 👋")

st.sidebar.success("Select an analysis from the sidebar to begin exploring music data.")

st.markdown(
    """
    Welcome to **Music dataGalax**, your go-to app for exploring and analyzing music data! 
    Dive into the world of music through data-driven insights and interactive visualizations. 
    Whether you're a music enthusiast, data analyst, or industry professional, our app offers a 
    variety of analyses tailored to uncover the stories behind the songs, artists, and trends 
    shaping the music world.

    ## Features
    - **Top Chart Analysis:** Dive into the most popular tracks and artists dominating the charts.
    - **Track Recommendation:** Discover new music based on your taste and preferences.
    - **Popularity Check:** Analyze the popularity trends of songs and artists over time.
    - **Track Comparison:** Compare the attributes of different tracks to see what makes a hit.
    - **Era Comparison:** Explore how musical trends and preferences have evolved across different eras.

    Ready to uncover the secrets of the music industry through data?  
    **👈 Select an analysis from the sidebar** and let the journey begin!

    ### Want to learn more about music data analysis?
    Whether you're new to the field or looking to expand your knowledge, check out these resources:
    - Delve into our [documentation](https://docs.yourapp.com) for detailed guides on using the app.
    - Join the conversation in our community forum (Coming Soon)

    ### Inspired to create your own data project?
    Streamlit is an open-source app framework perfect for building interactive and shareable web apps for data science and machine learning projects.
    - Get started with Streamlit at [streamlit.io](https://streamlit.io)
    - Jump into Streamlit [documentation](https://docs.streamlit.io)
"""
)

st.image('images/dataGalax.png', use_column_width=True, caption="Explore the rhythm of data.")