# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:16:28 2025

@author: William Liu
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def vis_dataset(df):
    # Scatterplot
    df['review_length'] = df['text_'].apply(len)
    sns.scatterplot(data=df, x='rating', y='review_length', hue='label', palette='coolwarm', alpha=0.5)
    plt.title('Rating vs. Review Length by Label')
    plt.show()
    
    # Pairplot
    plot_cols = ['rating', 'review_length']
    sns.pairplot(df, vars=plot_cols, hue='label', palette='coolwarm')
    plt.suptitle('Pairplot of Numeric Features by Label', y=1.02)
    plt.show()
    
    return
