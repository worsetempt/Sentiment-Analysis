# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:15:50 2025

@author: William Liu
"""

import numpy as np
import pandas as pd
import re
import nltk
import os
import kagglehub
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def loadData():
    # Download NLTK Resources 
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Download dataset
    path = kagglehub.dataset_download("mexwell/fake-reviews-dataset")
    
    csv_path = os.path.join(path, "fake reviews dataset.csv")
    
    # Load Dataset
    amazon_df = pd.read_csv(csv_path)  
    amazon_df = amazon_df[['text_', 'label', 'rating']]  # Keep only text and label columns
    
    # Map labels
    amazon_df['label'] = amazon_df['label'].map({'CG': 1, 'OR': 0})  # 1=Fake, 0=Real
    
    return amazon_df

def process_text(text):
    # Clean text
    text = re.sub(r'[^\w\s]', '', str(text))  # Handle NaN values with str()
    text = text.lower() # Lowercase
    text = re.sub(r'\d+', '', text) #Removes all digit sequences
    
    # Tokenize and lemmatize
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def preprocess():
    amazon_df = loadData()
    amazon_df['cleaned_text'] = amazon_df['text_'].apply(process_text)
    
    amazon_df.to_csv('preprocessed_data.csv', index=False)
    
    return amazon_df


