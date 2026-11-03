%%writefile app.py
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="IMDb Sentiment Classifier", page_icon="🎬")

st.title("🎬 IMDb Sentiment Analyzer")
st.write("This application uses a fine-tuned, optimized DistilBERT model hosted on the Hugging Face Hub to classify movie reviews.")

# Cache the pipeline resource so it doesn't reload on every button click
@st.cache_resource
def load_classifier():
    model_repo = "Zubair-Dashti/distilbert-imdb-sentiment-optimized"
    return pipeline("text-classification", model=model_repo)

try:
    with st.spinner("Loading optimized model from Hugging Face Hub..."):
        classifier = load_classifier()
    
    user_input = st.text_area("Enter your movie review here:", "This movie was fantastic! The acting was supreme.")
    
    if st.button("Classify Sentiment"):
        if user_input.strip() != "":
            result = classifier(user_input)[0]
            label = result['label']
            score = result['score']
            
            if label == "POSITIVE":
                st.success(f"✨ **Positive Sentiment** (Confidence: {score:.2%})")
            else:
                st.error(f"💔 **Negative Sentiment** (Confidence: {score:.2%})")
        else:
            st.warning("Please write some text before analyzing.")
except Exception as e:
    st.error(f"Error loading the model: {e}")
