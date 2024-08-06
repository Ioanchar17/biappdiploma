from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

clean_reviews_iphone = pd.read_csv('Clean_datasets/iphone_clean.csv')

clean_reviews_iphone = clean_reviews_iphone.dropna()

vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(2,2))
X = vectorizer.fit_transform(clean_reviews_iphone['cleaned_review'])

# Encode labels

le = LabelEncoder()
y = le.fit_transform(clean_reviews_iphone['rating'])

print(f'Original dataset shape : {Counter(y)}')

smote = SMOTE(random_state=2024)
X_res, y_res = smote.fit_resample(X, y)

print(f'Resampled dataset shape {Counter(y_res)}')

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, train_size=0.75, test_size=0.25,random_state=2024)


pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=2024, verbosity=2,config_dict = 'TPOT sparse')
pipeline_optimizer.fit(X_train, y_train)
pipeline_optimizer.export('tpot_exported_pipeline_iphone.py')

