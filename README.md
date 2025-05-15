# Sentiment Analysis of Product Reviews

## Overview

This project applies machine learning and natural language processing (NLP) techniques to classify Amazon product reviews as positive or negative. The goal is to help users quickly understand product feedback and assist businesses in monitoring customer sentiment.

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
- **Python Script**
  To train and test the model via command line:
    ```
    python src/main.py
    ```

- **Docker**  
  Install Docker and WSL and run via command line:
    ```
    docker build -t flask-app .
    docker run -p 5000:5000 flask-app
    ```
  Then try to predict a sentiment:
    ```
    curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"This product is amazing!\"}"
    ```
---

## Results

- **Accuracy:** XX%
- **Precision / Recall / F1 Score:** XX / XX / XX
- **Sample Output:**
    ```
    Review: "This product is amazing!"
    Predicted Sentiment: Positive
    ```

- *(Add confusion matrix or sample plots if available)*

---

## Future Work

- Test with advanced models (e.g., LSTM, BERT)
- Expand to multi-class sentiment (e.g., star ratings)

---

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.
