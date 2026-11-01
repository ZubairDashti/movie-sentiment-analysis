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

# Sample reviews selection outside the form
samples = {
    "Select a sample...": "",
    "Positive Sample 1": "This movie was absolutely amazing! Excellent performance and great story.",
    "Positive Sample 2": "A breathtaking masterpiece. Outstanding visual effects and direction.",
    "Negative Sample 1": "Extremely boring film. Waste of time and money, terrible acting.",
    "Negative Sample 2": "Awful storyline, poor pacing, and completely uninteresting characters."
}

selected_sample = st.selectbox("Choose a sample review:", list(samples.keys()))
default_text = samples[selected_sample] if selected_sample != "Select a sample..." else ""

# Form container isolates the user input and submission
with st.form(key="sentiment_form"):
    user_input = st.text_area("Review Text:", value=default_text, height=130)
    submit_button = st.form_submit_button(label="Analyze Sentiment")

# Model execution takes place ONLY when submit_button is True AND user_input is not empty
if submit_button:
    clean_text = user_input.strip()
    if clean_text:
        with st.spinner("Analyzing review..."):
            prediction = classifier(clean_text)[0]
            raw_label = prediction['label']
            score = prediction['score']
            
            # Label mapping
            if raw_label in ["LABEL_1", "POSITIVE"]:
                sentiment = "Positive 😃"
                st.success(f"**Result:** {sentiment}")
            else:
                sentiment = "Negative 😞"
                st.error(f"**Result:** {sentiment}")
                
            st.metric(label="Confidence Score", value=f"{score:.2%}")
    else:
        st.warning("Please enter or select a movie review before clicking analyze.")
