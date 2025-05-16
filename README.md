# Sentiment Analysis of Product Reviews

## Overview

This project applies machine learning and natural language processing (NLP) techniques to classify product reviews as real or fake. The goal is to help users quickly understand product feedback and assist businesses in monitoring customer sentiment. 

---

## Description
- Uses random search to find the best parameters for the TfidfVectorizer.
- Uses grid search to find the best model and its associated parameters (SVM, Random Forest, Naive Bayes).
- Uses grid search to find the best hyperparameters for XGBClassifier.
- Compares the evaluations for the selected model and XGBClassifier.
  
---

## Requirements

Python 3.11

## Dataset

- **Source:** [https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset]
- **Description:** The dataset contains reviews, each labeled as CG (1) or OR (0) sentiment.

---

## Installation

1. **Clone the repository:**
    ```
    git clone https://github.com/worsetempt/Sentiment-Analysis
    cd amazon-sentiment-analysis
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

---

## Usage
-   To visualize the dataset or train and test the models via command line:
    ```
    python src/main.py
    ```

-   If you have Docker installed, open command and navigate to project directory (where Dockerfile is located). Make sure deploy.py and model/vectorizer files are in same directory. Then run:
    ```
    docker build -t flask-app .
    ```
    After the image builds successfully, start your container and map the ports:
    ```
    docker run -p 5000:5000 flask-app
    ```
    Test the Flask API by predicting a sentiment in a separate cmd window, you should expect a fake review prediction (1) for this example:
    ```
    curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"This product is amazing!\"}"
    ```
---

## Results
**Selected best model: SVM**
- **Accuracy:** 91.3%
- **Precision / Recall / F1 Score:** 0.91 / 0.91 / 0.91
- **Confusion Matrix:** [[3752  319] [ 384 3632]]

**XGBClassifier**
- **Accuracy:** 88.7%
- **Precision / Recall / F1 Score:** 0.89 / 0.89 / 0.89
- **Confusion Matrix:** [[3693  378] [ 533 3483]]

**Sample Output:**
    ```
    Review: "Great product, I recommend!"
    Predicted Sentiment: 1
    ```
    
---

## Future Work

- Test with advanced models (e.g., LSTM, BERT)
- Expand to multi-class sentiment (e.g., star ratings)

---

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.
