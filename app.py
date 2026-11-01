import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Click a sample review or enter your own text to test the model.")

MODEL_REPO = "Zubair-Dashti/distilbert-imdb-sentiment"

@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model=MODEL_REPO)

classifier = load_pipeline()

# Sample reviews
samples = {
    "Select a sample...": "",
    "Positive Sample 1": "This movie was absolutely amazing! Excellent performance and great story.",
    "Positive Sample 2": "A breathtaking masterpiece. Outstanding visual effects and direction.",
    "Negative Sample 1": "Extremely boring film. Waste of time and money, terrible acting.",
    "Negative Sample 2": "Awful storyline, poor pacing, and completely uninteresting characters."
}

# Select sample review
selected_sample = st.selectbox("Choose a sample review:", list(samples.keys()))
default_text = samples[selected_sample] if selected_sample != "Select a sample..." else ""

# Input text box
user_input = st.text_area("Review Text:", value=default_text, height=130)

# Single Analyze button that triggers evaluation
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Get prediction from Hugging Face model
        prediction = classifier(user_input)[0]
        raw_label = prediction['label']
        score = prediction['score']
        
        # Map raw Hugging Face labels (LABEL_1/LABEL_0) to clear human readable output
        if raw_label in ["LABEL_1", "POSITIVE"]:
            sentiment = "Positive 😃"
            st.success(f"**Result:** {sentiment}")
        else:
            sentiment = "Negative 😞"
            st.error(f"**Result:** {sentiment}")
            
        st.metric(label="Confidence Score", value=f"{score:.2%}")
    else:
        st.warning("Please enter or select a movie review before clicking analyze.")
