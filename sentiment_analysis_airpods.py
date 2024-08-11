import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

clean_reviews_airpods = pd.read_csv('Clean_datasets/airpods_clean.csv')

clean_reviews_airpods = clean_reviews_airpods.dropna()

vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(2,2))
X = vectorizer.fit_transform(clean_reviews_airpods['cleaned_review'])

# Encode labels

le = LabelEncoder()
y = le.fit_transform(clean_reviews_airpods['rating'])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25,random_state=2024)


pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=2024, verbosity=2,config_dict = 'TPOT sparse')
pipeline_optimizer.fit(X_train, y_train)
pipeline_optimizer.export('tpot_exported_pipeline_airpods.py')