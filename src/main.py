# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:32:51 2025

@author: William Liu
"""

from preprocessing import preprocess
from visualization import vis_dataset
from model import train, train_xgb, vectorize
from evaluate import evaluate
import joblib

def main():
    amazon_df = preprocess()
    vis_dataset(amazon_df)
    X = amazon_df['cleaned_text']
    Y = amazon_df['label']
    
    X_train, X_test, Y_train, Y_test = vectorize(X, Y)
    
    #best_model, X_train, X_test, Y_train, Y_test = train(X, Y)
    #xgb_m, X_train, X_test, Y_train, Y_test = train_xgb(X, Y)
    
    #model = joblib.load('model.pkl')
    xgb_m = joblib.load('xgb_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
   # evaluate(model, X_test, Y_test)
    evaluate(xgb_m, X_test, Y_test)
    
    text = 'Great product, I recommend'
    X_new = vectorizer.transform([text])
    prediction = xgb_m.predict(X_new)
    print("Review: ", text)
    print ("Sentiment: ", prediction) # 1 for fake and 0 for real
    
    text = 'I recommend the product cuz its nice and comfy'
    X_new = vectorizer.transform([text])
    prediction = xgb_m.predict(X_new)
    print("Review: ", text)
    print ("Sentiment: ", prediction) # 1 for fake and 0 for real
    
if __name__ == "__main__":
    main()
