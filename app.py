import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬")

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

# Dropdown for selecting pre-made reviews
selected_sample = st.selectbox("Choose a sample review:", list(samples.keys()))

# Set initial text based on selection
default_text = samples[selected_sample] if selected_sample != "Select a sample..." else ""

# Input area
user_input = st.text_area("Review Text:", value=default_text, height=120)

if st.button("Analyze Sentiment") or (selected_sample != "Select a sample..."):
    if user_input.strip():
        result = classifier(user_input)[0]
        label = result['label']
        score = result['score']
        
        sentiment = "Positive 😃" if label in ["LABEL_1", "POSITIVE"] else "Negative 😞"
        
        st.subheader(f"Result: {sentiment}")
        st.write(f"**Confidence Score:** {score:.2%}")
    else:
        st.info("Please select a sample or enter text to analyze.")
