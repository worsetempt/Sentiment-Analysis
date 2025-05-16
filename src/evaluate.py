# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:05:26 2025

@author: bojvn
"""
from sklearn.metrics import classification_report, confusion_matrix
    
def evaluate(model, X_test, Y_test):
    print(f"\n{'='*50}")
    print("Model class name:", model.__class__.__name__)
    print("Model hyperparameters:")
    for param, value in model.get_params().items():
        print(f"{param}: {value}")
    
    test_accuracy = model.score(X_test, Y_test)
        
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))
    print(f"{'='*50}\n")
    
    return
