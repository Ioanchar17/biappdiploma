import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

clean_reviews_airpods = pd.read_csv('https://github.com/Ioanchar17/biappdiploma/blob'
                                    '/378f0955b61a238883ebdeb0d580209748ea1110/Clean_datasets/airpods_clean.csv')

clean_reviews_airpods = clean_reviews_airpods.dropna()

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
X = vectorizer.fit_transform(clean_reviews_airpods['cleaned_review'])

# Encode labels

le = LabelEncoder()
y = le.fit_transform(clean_reviews_airpods['rating'])


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=2024)

exported_pipeline = MultinomialNB(alpha=0.1, fit_prior=True)

exported_pipeline.fit(X_train, y_train)
predictions = exported_pipeline.predict(X_test)
acc_score = accuracy_score(y_test, predictions)

## Predict sentiment for dataset


clean_reviews_airpods['predicted_sentiment'] = exported_pipeline.predict(X)

print(acc_score)
