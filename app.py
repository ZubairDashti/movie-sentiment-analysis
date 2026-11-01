import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to test the sentiment model.")

MODEL_REPO = "Zubair-Dashti/distilbert-imdb-sentiment"

@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model=MODEL_REPO)

# Initialize classifier
classifier = load_pipeline()

# Initialize session state variable to strictly control display
if "has_analyzed" not in st.session_state:
    st.session_state.has_analyzed = False

# Form container ensures inputs do not trigger rerun logic prematurely
with st.form(key="review_form"):
    user_input = st.text_area(
        "Review Text:", 
        height=140, 
        placeholder="Type or paste your movie review here..."
    )
    submit_button = st.form_submit_button(label="Analyze Sentiment")

# Execution ONLY happens when form is submitted
if submit_button:
    clean_text = user_input.strip()
    if clean_text:
        st.session_state.has_analyzed = True
        st.session_state.analyzed_text = clean_text
    else:
        st.session_state.has_analyzed = False
        st.warning("Please enter a review before clicking analyze.")

# Render output ONLY if a valid analysis took place
if st.session_state.has_analyzed:
    with st.spinner("Classifying sentiment..."):
        prediction = classifier(st.session_state.analyzed_text)[0]
        raw_label = prediction['label']
        score = prediction['score']
        
        # Label mapping for DistilBERT
        if raw_label in ["LABEL_1", "POSITIVE"]:
            st.success("**Result:** Positive 😃")
        else:
            st.error("**Result:** Negative 😞")
        
        st.metric(label="Confidence Score", value=f"{score:.2%}")
