import streamlit as st
import pickle
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from lime.lime_text import LimeTextExplainer

# Download nltk data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 AI Email/SMS Spam Detector")

st.write("Enter a message and the model will classify it as Spam or Not Spam.")

ps = PorterStemmer()

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Predictor function for LIME
def predictor(texts):
    transformed = [transform_text(t) for t in texts]
    vectorized = tfidf.transform(transformed)
    return model.predict_proba(vectorized)

# Input
input_sms = st.text_area("Enter the message")

# Button
if st.button("Predict"):

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    
    prob = model.predict_proba(vector_input)

    spam_prob = prob[0][1]

    st.write("### Prediction Result")

    if result == 1:
        st.error("Spam Message 🚨")
    else:
        st.success("Not Spam ✅")

    st.write(f"Spam Probability: **{spam_prob*100:.2f}%**")

    st.progress(int(spam_prob*100))

    # LIME explanation
    st.write("## Important Words (LIME Explanation)")

    explainer = LimeTextExplainer(class_names=["Not Spam","Spam"])

    exp = explainer.explain_instance(transform_text(input_sms), predictor, num_features=6)
    lime_list = exp.as_list()
    # Display words
    if len(lime_list) == 0:
        st.warning("No important words detected.")
    else:
        for word, score in lime_list:
            if score > 0:
                st.write(f"🟥 **{word}** → increases Spam probability ({score:.3f})")
            else:
                st.write(f"🟩 **{word}** → decreases Spam probability ({abs(score):.3f})")

    # Chart visualization
    import matplotlib.pyplot as plt

    words = [i[0] for i in lime_list]
    scores = [i[1] for i in lime_list]
    
    colors = ["red" if s > 0 else "green" for s in scores]
    
    fig, ax = plt.subplots()
    ax.barh(words, scores, color=colors)
    
    ax.set_xlabel("Word Impact")
    ax.set_title("LIME Explanation")
    
    st.pyplot(fig)
