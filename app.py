import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to test the sentiment model.")

MODEL_REPO = "Zubair-Dashti/distilbert-imdb-sentiment"

@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model=MODEL_REPO)

classifier = load_pipeline()

# Single text input box — NO sample selector
user_input = st.text_area(
    "Review Text:", 
    height=140, 
    placeholder="Type or paste your movie review here..."
)

# Analyze button
if st.button("Analyze Sentiment"):
    clean_text = user_input.strip()
    
    if clean_text:
        with st.spinner("Classifying sentiment..."):
            prediction = classifier(clean_text)[0]
            raw_label = prediction['label']
            score = prediction['score']
            
            # Label mapping for DistilBERT
            if raw_label in ["LABEL_1", "POSITIVE"]:
                st.success("**Result:** Positive 😃")
            else:
                st.error("**Result:** Negative 😞")
            
            st.metric(label="Confidence Score", value=f"{score:.2%}")
    else:
        st.warning("Please enter a review before clicking analyze.")
