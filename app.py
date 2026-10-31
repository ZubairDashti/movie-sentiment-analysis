import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to classify whether the review is Positive or Negative.")

# Replace with your uploaded Hugging Face repo path
MODEL_REPO = "YOUR_USERNAME/distilbert-imdb-sentiment"

@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model=MODEL_REPO)

classifier = load_pipeline()

user_input = st.text_area("Review Text:", "The cinematography was stunning and the acting was top-tier. Highly recommended!")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = classifier(user_input)[0]
        label = result['label']
        score = result['score']

        # DistilBERT default outputs standard labels (e.g., LABEL_1=Positive, LABEL_0=Negative)
        sentiment = "Positive 😃" if label in ["LABEL_1", "POSITIVE"] else "Negative 😞"

        st.subheader(f"Result: {sentiment}")
        st.write(f"**Confidence Score:** {score:.2%}")
    else:
        st.warning("Please enter text to analyze.")
