import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

clean_reviews = pd.read_csv('Clean_datasets\mac_clean.csv')

clean_reviews = clean_reviews.dropna()

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
X = vectorizer.fit_transform(clean_reviews['cleaned_review'])

# Encode labels

le = LabelEncoder()
y = le.fit_transform(clean_reviews['rating'])

print(f'Original dataset shape : {Counter(y)}')

smote = SMOTE(random_state=2024)
X_res, y_res = smote.fit_resample(X, y)

print(f'Resampled dataset shape {Counter(y_res)}')

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, train_size=0.75, test_size=0.25, random_state=2024)

# Average CV score on the training set was: 0.942145099686044
exported_pipeline = BernoulliNB(alpha=0.001, fit_prior=True)

exported_pipeline.fit(X_train, y_train)
predictions = exported_pipeline.predict(X_test)
acc_score = accuracy_score(y_test, predictions)

## Predict sentiment for dataset


clean_reviews['predicted_sentiment'] = exported_pipeline.predict(X)

print(acc_score)




