import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import cmudict
from collections import defaultdict
from transformers import pipeline, AutoTokenizer
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lyric Sentiment Analysis", page_icon="🖋️")

# Download NLTK data files (you only need to do this once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('cmudict')

class SentimentAnalysis:
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        self.emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
        self.max_length = self.tokenizer.model_max_length
        self.chunk_size = self.max_length - 2  # Account for special tokens 
        self.emotion_to_emoji = {
            'anger': '😡',
            'love': '🥰',
            'fear': '😨',
            'joy': '😊',
            'sadness': '😢',
            'surprise': '😲'
        }
        self.cmu_dict = nltk.corpus.cmudict.dict()

    def preprocess_lyrics(self, lyrics):
        # Convert to lowercase
        lyrics = lyrics.lower()

        # Remove URLs
        lyrics = re.sub(r'http\S+|www\S+|https\S+', '', lyrics)

        # Remove special tags like [VERSE], [CHORUS], etc.
        lyrics = re.sub(r'\[.*?\]', '', lyrics)

        # Remove special characters 
        lyrics = re.sub(r'[^a-zA-Z0-9\s]', '', lyrics)

        # Tokenize the lyrics
        words = nltk.word_tokenize(lyrics)

        # Remove stopwords
        cleaned_words = [word for word in words if word not in self.stop_words]

        # Join words back into a single string
        cleaned_lyrics = ' '.join(cleaned_words)

        return cleaned_lyrics
    
    def unique_words(self, lyrics):
        cleaned_lyrics = self.preprocess_lyrics(lyrics)
        words = cleaned_lyrics.split()
        unique_words = set(words)
        total_unique_words = len(unique_words)
        return total_unique_words

    def frequency_words(self, lyrics):
        cleaned_lyrics = self.preprocess_lyrics(lyrics)
        words = cleaned_lyrics.split()
        
        # Calculate word frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Convert word frequencies to DataFrame
        word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])

        # Sort by frequency and select top 20 words
        top_words_df = word_freq_df.sort_values(by='Frequency', ascending=False).head(20)

        # Plot using Plotly
        fig = px.bar(top_words_df, x='Word', y='Frequency', orientation='v',
                     title='Top 20 Most Frequent Words', labels={'Word': 'Word', 'Frequency': 'Frequency'})
        return fig
    
    def wordcloud(self, lyrics):
        cleaned_lyrics = self.preprocess_lyrics(lyrics)
        words = cleaned_lyrics.split()
        
        # Calculate word frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Generate WordCloud
        wordcloud = WordCloud(width=800, 
                              height=500, 
                              background_color='white',
                              colormap='plasma').generate_from_frequencies(word_freq)

        # Display WordCloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        return plt


    def classify_emotion(self, lyrics):
        # Preprocess the lyrics
        cleaned_lyrics = self.preprocess_lyrics(lyrics)
        print(f"Preprocessed: {cleaned_lyrics}")

        # Tokenize the cleaned lyrics
        tokens = self.tokenizer(cleaned_lyrics, return_tensors='tf', truncation=True, padding=True)
        input_ids = tokens['input_ids'][0]
        
        # Initialize emotion scores
        emotions = defaultdict(float)

        # Split input into chunks with context
        start_idx = 0
        while start_idx < len(input_ids):
            # Determine the end index of the current chunk
            end_idx = min(start_idx + self.chunk_size, len(input_ids))

            # Extract the current chunk
            chunk = input_ids[start_idx:end_idx]

            # Decode the chunk back to text
            chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
            print("Chunk text:", chunk_text)  # debugging

            # Perform emotion classification on the chunk
            try:
                results = self.emotion_classifier(chunk_text)
                print("Results:", results)  # debugging

                # Check if results is a list
                if isinstance(results, list):
                    # Iterate over each inner list in results
                    for inner_list in results:
                        # Check if inner_list is a list of dictionaries
                        if isinstance(inner_list, list) and all(isinstance(item, dict) for item in inner_list):
                            # Accumulate emotion scores from each dictionary in inner_list
                            for result in inner_list:
                                label = result['label']
                                score = result['score']
                                emotions[label] += score
                        else:
                            print("Unexpected inner list format:", inner_list)  # Added for debugging
                else:
                    print("Unexpected results format:", results)  # Added for debugging
            except Exception as e:
                print("Error during emotion classification:", e)  # Added for debugging

            # Move to the next chunk
            start_idx += self.chunk_size  # Move to the next non-overlapping chunk

        # Normalize emotion scores
        total_score = sum(emotions.values())
        print(f"total score: {total_score}")
        for emotion in emotions:
            emotions[emotion] /= total_score

        return dict(emotions)
    
    def visualize_rhymes(self, lyrics):
        def phonetic_transcription(lyrics):
            # Preprocess lyrics
            cleaned_lyrics = self.preprocess_lyrics(lyrics)
            words = cleaned_lyrics.split()

            # Get phonetic transcriptions
            phonetic_lyrics = []
            for word in words:
                if word in self.cmu_dict:
                    phonetic_lyrics.append(self.cmu_dict[word][0])

            return phonetic_lyrics

        def analyze_rhymes(phonetic_lyrics):
            # Analyze rhymes
            rhymes = {}
            for phoneme in phonetic_lyrics:
                last_phoneme = phoneme[-1]
                if last_phoneme in rhymes:
                    rhymes[last_phoneme] += 1
                else:
                    rhymes[last_phoneme] = 1

            return rhymes

        # Get phonetic transcriptions
        phonetic_lyrics = phonetic_transcription(lyrics)

        # Analyze rhymes
        rhymes = analyze_rhymes(phonetic_lyrics)

        # Create DataFrame for visualization
        rhymes_df = pd.DataFrame(list(rhymes.items()), columns=['Phoneme', 'Frequency'])
        rhymes_df = rhymes_df.sort_values(by='Frequency', ascending=False).head(20)

        # Plot
        fig = px.bar(
            rhymes_df,
            x='Phoneme',
            y='Frequency',
            title='Top 20 Most Frequent Rhyme Phonemes',
            labels={'Phoneme': 'Phoneme', 'Frequency': 'Frequency'}
        )
        return fig

    # Run All Analysis
    def run_analysis(self, lyrics):
        st.header("Unique Words")
        unique_words = self.unique_words(lyrics)
        st.write(f"### Total unique words: {unique_words}")

        st.header("Frequency of Words")
        fig = self.frequency_words(lyrics)
        st.plotly_chart(fig)

        st.header("Keyword Cloud")
        wordcloud = self.wordcloud(lyrics)
        st.pyplot(wordcloud)

        st.header("Emotion Classification Results")
        emotions = sa.classify_emotion(lyrics)
        sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
        for emotion, score in sorted_emotions:
            emoji = sa.emotion_to_emoji.get(emotion, '🤷')
            st.write(f"### {emoji} {emotion.capitalize()}: {score:.2f}")
            
        st.header("Rhyme Analysis")
        fig = sa.visualize_rhymes(lyrics)
        st.plotly_chart(fig)

# Main Section
if "lyrics_input" not in st.session_state:
    st.session_state.lyrics_input = ""

# Clear text callback
def clear_text():
    st.session_state.lyrics_input = ""

st.title("Lyric Sentiment Analysis")
st.info("This page allows users to analyze the sentiment of song lyrics. Due to copyright restrictions, the app does not fetch lyric data from APIs. Instead, users can simply copy and paste their favorite lyrics into the input box to perform the analysis. For lyric references, we recommend using websites such as https://app.lyricsondemand.com/.")

# Input box
lyrics_input = st.text_area("Enter song lyrics here", height=200, key="lyrics_input")
analyze_button = st.button("Analyze")
clear_button = st.button("Clear", on_click=clear_text)


if analyze_button:
    if lyrics_input:
        sa = SentimentAnalysis()
        try:
            
            sa.run_analysis(lyrics_input)
            st.balloons()

        except Exception as e:
            st.error(f"Error occurred while analyzing the lyrics: {e}")

    else:
        st.error("Please enter lyrics to analyze.")

