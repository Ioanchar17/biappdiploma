import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

clean_reviews_vader = pd.read_csv('Clean_datasets\mac_clean.csv')

clean_reviews_vader = clean_reviews_vader.dropna()

# creating object
sentiments = SentimentIntensityAnalyzer()

clean_reviews_vader['compound'] = [sentiments.polarity_scores(i)["compound"] for i in
                                   clean_reviews_vader["cleaned_review"]]

score = clean_reviews_vader["compound"].values
sentiment = []
for i in score:
    if i >= 0.05:
        sentiment.append('positive')
    elif i <= -0.05:
        sentiment.append('negative')
    else:
        sentiment.append('neutral')
clean_reviews_vader["sentiment"] = sentiment

print(clean_reviews_vader.sentiment.value_counts())