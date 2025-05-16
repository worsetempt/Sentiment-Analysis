# -*- coding: utf-8 -*-
"""
Created on Thu May 15 17:12:11 2025

@author: William Liu
"""

from flask import Flask, request, jsonify
import joblib

# Load model and vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():
    data = request.json
    text = data['text']
    
    # Transform text
    X_new = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(X_new)[0]
    
    return jsonify({
        'text': text,
        'sentiment': int(prediction)  # Assuming 0=Real, 1=Fake
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


    