# 📩 AI Spam Detection System

An **Explainable Machine Learning Web Application** that detects whether an **Email/SMS message is Spam or Not Spam**.
The system uses **TF-IDF feature extraction, a trained ML classifier, and LIME explainability** to show **why the model made a prediction**.

The application is built with an interactive interface using **Streamlit**.

---

# 🚀 Live Application

```
 https://spamdetection-z2txqoyhbupqjgajdcdzut.streamlit.app/
```

---

# 🧠 Project Overview

Spam messages are a major problem in digital communication.
This project builds a **Machine Learning based spam detection system** that:

* Classifies messages as **Spam or Not Spam**
* Shows **prediction probability**
* Explains the decision using **LIME (Explainable AI)**

This improves **model transparency and user trust**.

---

# ✨ Features

✔ Spam / Not Spam classification
✔ Text preprocessing using **NLTK**
✔ **TF-IDF vectorization** for text features
✔ Machine Learning classification
✔ **LIME explanation of predictions**
✔ Spam probability meter
✔ Interactive **Streamlit UI**

---

# 🛠️ Technologies Used

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Programming language |
| Streamlit    | Web interface        |
| Scikit-learn | Machine learning     |
| NLTK         | Text preprocessing   |
| TF-IDF       | Feature extraction   |
| LIME         | Explainable AI       |
| Matplotlib   | Visualization        |

---

# 📂 Project Structure

```
SpamDetection
│
├── app.py              # Streamlit application
├── train_model.py      # Model training script
├── model.pkl           # Saved trained model
├── vectorizer.pkl      # Saved TF-IDF vectorizer
├── requirements.txt    # Project dependencies
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Shourabh8/SpamDetection.git
```

Move into the project folder:

```
cd SpamDetection
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

---

# 🧪 How the System Works

The spam detection process follows these steps:

```
User Message
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Machine Learning Model
     ↓
Spam / Not Spam Prediction
     ↓
LIME Explanation
```

---

# 🔍 Example Prediction

### Input Message

```
Congratulations! You have won a free lottery.
Click here to claim your prize.
```

### Output

```
Prediction: Spam 🚨
Spam Probability: 92%
```

### Important Words Detected

```
free
lottery
won
claim
```

These words increase the likelihood of the message being classified as spam.

---

# 🧠 Explainable AI (LIME)

The project uses **LIME (Local Interpretable Model-agnostic Explanations)** to show **which words influenced the model’s decision**.

Example explanation:

```
free → increases spam probability
lottery → increases spam probability
claim → increases spam probability
```

This makes the model **transparent and interpretable**.

---

# 📊 Machine Learning Model

The spam detection model uses:

* **TF-IDF Vectorizer** for feature extraction
* **Multinomial Naive Bayes classifier**

Why Naive Bayes?

✔ Works very well for text classification
✔ Fast and efficient
✔ Performs well on spam datasets

---

# 📈 Future Improvements

Possible upgrades for the project:

* Deep Learning spam detection
* Email inbox integration
* REST API for spam detection
* Larger dataset training
* Advanced visualization dashboard

---

# 📸 Application Screenshot
Example:

```
 ### Prediction Interface
<p align="center">
  <img src="images/app_ui.png" width="800">
</p>
### Spam Probability Meter
<p align="center">
  <img src="images/Prediction_output.png" width="800">
</p>
### LIME Explanation Graph
<p align="center">
  <img src="images/Lime_output.png" width="800">
</p>
```

---

# 👨‍💻 Author

**Shourabh Jamwal**

Machine Learning Enthusiast

---

# ⭐ If you like this project

Please consider **starring the repository** on GitHub.
