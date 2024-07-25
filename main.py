from flask import Flask, request, jsonify, render_template_string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
import os


app = Flask(__name__)

model = load_model(r'H:\Chain Code Consulting\Movie Sentiment Analysis\sentiment_model (1).h5')

tokenizer_path = r'H:\Chain Code Consulting\Movie Sentiment Analysis\tokenizer (1).pickle'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)


def preprocess_text(text):

    text = text.lower()

    text = ''.join(char for char in text if char.isalpha() or char.isspace())

    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(word) for word in text.split())
    return text


# HTML template
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
</head>
<body>
    <h1>Movie Review Sentiment Analysis</h1>
    <form action="/predict" method="post">
        <textarea name="review" rows="4" cols="50" placeholder="Enter your movie review here..."></textarea><br><br>
        <input type="submit" value="Analyze Sentiment">
    </form>
    {% if sentiment %}
    <h2>Sentiment: {{ sentiment }}</h2>
    {% endif %}
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['review']
    

    processed_text = preprocess_text(text)
    
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    
    # Predicting sentiment
    prediction = model.predict(padded_sequence)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    
    return render_template_string(html_template, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
