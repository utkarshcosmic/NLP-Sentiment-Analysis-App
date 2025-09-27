import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
port_stem = PorterStemmer()

# Load your trained model
model = pickle.load(open("/Users/utkarsh_verma/Codes/VS CODE/Sentiment_analysis_project/sentiment_analysis_model", "rb"))
vectorizer = pickle.load(open("/Users/utkarsh_verma/Codes/VS CODE/Sentiment_analysis_project/vectorizer.pkl", "rb"))

# Function to preprocess text
def clean_text(text):
  text = re.sub('[^a-zA-Z]',' ',text)
  text = text.lower()
  text = text.split()
  text = [port_stem.stem(word) for word in text if not word in stopwords.words('english')]
  text = ' '.join(text)
  return text

# Streamlit UI
st.title("Twitter Sentiment Analysis 🚀")
st.write("Enter a tweet to analyze its sentiment:")

user_input = st.text_area("Tweet text here...")

if st.button("Analyze"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("Positive 😃")
        else:
            st.error("Negative 😠")
    else:
        st.warning("Please enter some text to analyze.")