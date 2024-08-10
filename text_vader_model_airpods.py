import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

clean_reviews_vader_airpods = pd.read_csv('Clean_datasets/airpods_clean.csv')

clean_reviews_vader_airpods = clean_reviews_vader_airpods.dropna()

# creating object
sentiments = SentimentIntensityAnalyzer()

clean_reviews_vader_airpods['compound'] = [sentiments.polarity_scores(i)["compound"] for i in
                                           clean_reviews_vader_airpods["cleaned_review"]]

score = clean_reviews_vader_airpods["compound"].values
sentiment = []
for i in score:
    if i >= 0.05:
        sentiment.append('positive')
    else:
        sentiment.append('negative')
clean_reviews_vader_airpods["sentiment"] = sentiment

print(clean_reviews_vader_airpods.sentiment.value_counts())