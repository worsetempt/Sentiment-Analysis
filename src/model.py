# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:15:58 2025

@author: William Liu
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from xgboost import XGBClassifier

def tfidf_randsearch(X, Y):
    # Define pipeline
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('temp_clf', MultinomialNB())
    ])

    # Parameter distribution for random search
    tfidf_param_dist = {
    'tfidf__max_features': np.arange(1000, 10001, 1000),
    'tfidf__ngram_range': [(1,1), (1,2)]
    }

    # Run randomized search
    random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=tfidf_param_dist,
    n_iter=6,
    cv=3,
    scoring='accuracy',
    random_state=42,
    verbose=10
    )
    random_search.fit(X, Y)
    
    print("Best TF-IDF parameters:", random_search.best_params_)

    best_tfidf_params = random_search.best_estimator_.named_steps['tfidf'].get_params()
    
    vectorizer = TfidfVectorizer(**best_tfidf_params)
    X_new = vectorizer.fit_transform(X)
    
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    
    return X_new

def split_data(X, Y):
    # Split data into train/test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X_train, X_test, Y_train, Y_test

def model_gridsearch(X_train, Y_train):
    pipeline = Pipeline([
        ('clf', SVC())  # Placeholder classifier
    ])
    
    param_grid = [
        {  # SVM 
            'clf': [SVC()],
            'clf__C': [0.1, 1, 10],  # Wider range including smaller C
            'clf__kernel': ['linear','rbf'],
            'clf__gamma': ['scale', 'auto']  # Added gamma tuning
        },
        {  # Random Forest
            'clf': [RandomForestClassifier()],
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [10, 20, None],  # Added lower depth
            'clf__min_samples_split': [2, 5]  # Added split control
        },
        {  # Naive Bayes 
            'clf': [MultinomialNB()],
            'clf__alpha': [0.01, 0.1, 0.5, 1, 2]
        }
    ]
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=2, 
        verbose=10, 
        scoring='accuracy',
        return_train_score=True  # Overfitting check
    )
    
    grid_search.fit(X_train, Y_train)
    
    # Get best model and scores
    best_model = grid_search.best_estimator_
    results = pd.DataFrame(grid_search.cv_results_)
    best_idx = grid_search.best_index_
    
    # Print overfitting diagnostics
    train_score = results.loc[best_idx, 'mean_train_score']
    val_score = results.loc[best_idx, 'mean_test_score']
    gap = train_score - val_score
    overfit_warning = "Warning: Large train-val gap!" if gap > 0.15 else ""
    
    print(f"\n{'='*50}")
    print(f"Best classifier: {type(best_model.named_steps['clf']).__name__}")
    print(f"Best params: {grid_search.best_params_}")
    print(f"Train score: {train_score:.3f}")
    print(f"Val score:   {val_score:.3f}")
    print(f"Gap:         {gap:.3f} {overfit_warning}")
    print(f"{'='*50}\n")
    
    joblib.dump(best_model, 'model.pkl')
    
    return best_model

def xgboost_gridsearch(X_train, Y_train):
    param_grid = {
    'max_depth': [5, 7],
    'learning_rate': [0.1, 0.2, 0.3],
    'n_estimators': [200, 300, 400],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'gamma': [0, 0.1, 0.2]
    }   
    
    xgb_model = XGBClassifier()  # or XGBClassifier for classification
    grid_search = GridSearchCV(
        estimator = xgb_model,
        param_grid = param_grid,
        cv=2,  
        scoring = 'accuracy',  
        verbose = 10,
        return_train_score=True  # Overfitting check
    )
    
    grid_search.fit(X_train, Y_train)
    results = pd.DataFrame(grid_search.cv_results_)
    best_idx = grid_search.best_index_    
    xgb_m = grid_search.best_estimator_
    
    train_score = results.loc[best_idx, 'mean_train_score']
    val_score = results.loc[best_idx, 'mean_test_score']
    gap = train_score - val_score
    
    print(f"\n{'='*50}")
    print(f"{type(xgb_m).__name__}")
    print(f"Best params: {grid_search.best_params_}")
    print(f"Train score: {train_score:.3f}")
    print(f"Val score:   {val_score:.3f}")
    print(f"Gap:         {gap:.3f}")
    print(f"{'='*50}\n")

    joblib.dump(xgb_m, 'xgb_model.pkl')

    return xgb_m

def train(X, Y):
    X_new = tfidf_randsearch(X, Y)
    X_train, X_test, Y_train, Y_test = split_data(X_new, Y)
    best_model = model_gridsearch(X_train, Y_train)
    
    return best_model, X_train, X_test, Y_train, Y_test

def train_xgb(X, Y):
    X_train, X_test, Y_train, Y_test = vectorize(X, Y)
    xgb_m = xgboost_gridsearch(X_train, Y_train)

    return xgb_m, X_train, X_test, Y_train, Y_test

def vectorize(X, Y):
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    X_new = vectorizer.fit_transform(X)
    X_train, X_test, Y_train, Y_test = split_data(X_new, Y)
    
    return X_train, X_test, Y_train, Y_test
    