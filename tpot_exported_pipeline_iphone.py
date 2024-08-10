import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score

clean_reviews_iphone = pd.read_csv('Clean_datasets\iphone_clean.csv')

clean_reviews_iphone = clean_reviews_iphone.dropna()

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
X = vectorizer.fit_transform(clean_reviews_iphone['cleaned_review'])

# Encode labels

le = LabelEncoder()
y = le.fit_transform(clean_reviews_iphone['rating'])

print(f'Original dataset shape : {Counter(y)}')

smote = SMOTE(random_state=2024)
X_res, y_res = smote.fit_resample(X, y)

print(f'Resampled dataset shape {Counter(y_res)}')

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, train_size=0.75, test_size=0.25, random_state=2024)

exported_pipeline = LinearSVC(C=20.0, dual=False, loss='squared_hinge', penalty='l2', tol=0.01)

exported_pipeline.fit(X_train, y_train)
predictions = exported_pipeline.predict(X_test)
acc_score = accuracy_score(y_test, predictions)

## Predict sentiment for dataset


clean_reviews_iphone['predicted_sentiment'] = exported_pipeline.predict(X)

print(acc_score)
