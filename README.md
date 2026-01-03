# Fake_News_Detection_NLP

## Project Overview

Social media and online platforms spread misinformation quickly. This project classifies news articles as FAKE or REAL by learning patterns from:

- Article title
- Article text

Both are vectorized separately and then combined into a single feature set to improve prediction quality.

## Tech Stack

- Python
- Pandas, NumPy
- NLTK (stopwords, stemming)
- Scikit-learn (SVM, Decision Tree, Random Forest, Logistic Regression, GridSearchCV)
- Matplotlib
- WordCloud
- XGBoost

## Dataset

Loaded from a Kaggle .csv file (news.csv)

Contains columns: title, text, label (REAL/FAKE)

In this notebook, the first 3000 samples are used for training/testing

Pipeline
1) Data Preprocessing

For both title and text:

Removed special characters using regex

Converted to lowercase

Tokenized into words

Removed stopwords (NLTK)

Applied stemming using PorterStemmer

Creates:

corpus → cleaned article text

corpus2 → cleaned article title

2) Visualization (WordCloud)

Generated word clouds to visualize the most frequent words in:

Titles

Text bodies

This helps understand dominant tokens and common patterns in the dataset.

3) Feature Extraction (Bag of Words)

Used CountVectorizer to transform:

Title corpus → news_title

Text corpus → news_body

Then combined both vectors:

final_vector = [title_features | text_features]
shape = (3000, 37533)

4) Train/Test Split

Split data into:

75% training

25% testing

random_state = 13

Models Trained & Results

The following models were trained and evaluated using:

Confusion Matrix

Precision / Recall / F1-score

Accuracy

Model	Accuracy (approx.)
SVM (sigmoid kernel)	~76.7%
Decision Tree (entropy)	~79.5%
Random Forest (n=65)	~87.1%
XGBoost	~88.1%
Logistic Regression	~90.3%
Logistic Regression + GridSearchCV	~90.4%

✅ Best observed performance: Logistic Regression (with tuning) ~ 90.4%

Hyperparameter Tuning
Random Forest (GridSearchCV)

Tuned n_estimators:

Best found: n_estimators = 50

Accuracy stayed around ~87%

Logistic Regression (GridSearchCV)

Tuned:

penalty = [l1, l2]

C over log space

solver = liblinear

Best params:

penalty: l2

C ≈ 0.0336

solver: liblinear

✅ Improved stability and slightly improved accuracy.

Predictions (Demo)

The notebook includes prediction examples on:

A known fake news sample → predicted FAKE

A known real news sample → predicted REAL

Prediction works by:

Cleaning & stemming input title + text

Vectorizing using the same Bag-of-Words approach

Combining vectors

Running trained classifier prediction

How to Run
1) Install dependencies
pip install numpy pandas matplotlib scikit-learn nltk wordcloud xgboost

2) Download NLTK stopwords

Run once:

import nltk
nltk.download('stopwords')

3) Run the notebook / script

Place news.csv in the project directory

Run the notebook top-to-bottom

Repository Structure (Suggested)
Fake_News_Detection_NLP/
├── Fake News Detection.ipynb
├── news.csv
├── README.md
└── requirements.txt

Future Improvements

Use TF-IDF instead of CountVectorizer

Try n-grams (bigrams/trigrams)

Add lemmatization (WordNetLemmatizer)

Use cross-validation + better tuning search space

Try transformer models (BERT/RoBERTa) for higher accuracy

Package prediction as a simple web app (Streamlit/Flask)
