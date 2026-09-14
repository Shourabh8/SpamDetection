import streamlit as st
import pickle
import string
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from lime.lime_text import LimeTextExplainer

# Page Configuration
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

# Download necessary NLTK corpora safely
@st.cache_resource
def setup_nltk():
    for pkg in ['punkt', 'stopwords', 'punkt_tab']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

setup_nltk()

st.title("📩 AI Email/SMS Spam Detector")
st.write("Enter a message below to classify it as **Spam** or **Not Spam** with Explainable AI (LIME).")

ps = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))
PUNCTUATION = set(string.punctuation)

# Fast text preprocessing
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    # Filter alphanumeric and remove stopwords / punctuation
    filtered = [
        word for word in tokens 
        if word.isalnum() and word not in STOP_WORDS and word not in PUNCTUATION
    ]
    
    # Stem words
    stemmed = [ps.stem(word) for word in filtered]
    return " ".join(stemmed)

# Cache model and vectorizer loading
@st.cache_resource
def load_artifacts():
    with open("vectorizer.pkl", "rb") as f_vec:
        tfidf = pickle.load(f_vec)
    with open("model.pkl", "rb") as f_model:
        model = pickle.load(f_model)
    return tfidf, model

tfidf, model = load_artifacts()

# Predictor function for LIME
def predictor(texts):
    transformed = [transform_text(t) for t in texts]
    vectorized = tfidf.transform(transformed)
    return model.predict_proba(vectorized)

# Input
input_sms = st.text_area("Enter the message:", placeholder="e.g. Congratulations! You've won a $1,000 gift card. Call now to claim.", height=120)

# Button
if st.button("🔍 Predict", type="primary", use_container_width=True):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        transformed_sms = transform_text(input_sms)
        
        if not transformed_sms.strip():
            st.info("ℹ️ The message only contains stopwords, numbers, or symbols. Model default: **Not Spam ✅**")
        else:
            with st.spinner("Analyzing message with Machine Learning & LIME..."):
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]
                prob = model.predict_proba(vector_input)
                spam_prob = float(prob[0][1])

                st.write("---")
                st.write("### 📊 Prediction Result")

                if result == 1:
                    st.error(f"🚨 **Spam Message Detected** (Confidence: {spam_prob * 100:.1f}%)")
                else:
                    st.success(f"✅ **Not Spam / Legitimate Message** (Confidence: {(1 - spam_prob) * 100:.1f}%)")

                st.write(f"Spam Probability: **{spam_prob * 100:.2f}%**")
                st.progress(min(max(int(spam_prob * 100), 0), 100))

                # LIME explanation
                st.write("---")
                st.write("### 🧠 Important Words (LIME Explainability)")
                st.caption("Words in red push the prediction towards **Spam**, while green words push towards **Not Spam**.")

                try:
                    explainer = LimeTextExplainer(class_names=["Not Spam", "Spam"])
                    num_words = len(transformed_sms.split())
                    features_to_show = min(6, num_words)
                    
                    exp = explainer.explain_instance(
                        transformed_sms, 
                        predictor, 
                        num_features=features_to_show,
                        num_samples=300
                    )
                    lime_list = exp.as_list()

                    if not lime_list:
                        st.info("No dominant keyword weights detected for this text.")
                    else:
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown("#### Word Impact List")
                            for word, score in lime_list:
                                if score > 0:
                                    st.write(f"🟥 **{word}** → increases Spam probability (+{score:.3f})")
                                else:
                                    st.write(f"🟩 **{word}** → decreases Spam probability ({score:.3f})")

                        with col2:
                            # Matplotlib chart
                            words = [i[0] for i in lime_list]
                            scores = [i[1] for i in lime_list]
                            colors = ["#e74c3c" if s > 0 else "#2ecc71" for s in scores]

                            fig, ax = plt.subplots(figsize=(6, 3.5))
                            ax.barh(words, scores, color=colors)
                            ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
                            ax.set_xlabel("Word Weight Impact")
                            ax.set_title("LIME Feature Attribution")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                except Exception as e:
                    st.info(f"LIME visualization note: Prediction completed, explanation skipped ({e}).")

