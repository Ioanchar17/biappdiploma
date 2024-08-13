import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

clean_reviews_vader_iphone = pd.read_csv('https://github.com/Ioanchar17/biappdiploma/blob'
                                         '/61c0c0b752f901e3f371a0998893429176f05caa/Clean_datasets/iphone_clean.csv')

clean_reviews_vader_iphone = clean_reviews_vader_iphone.dropna()

# creating object
sentiments = SentimentIntensityAnalyzer()

clean_reviews_vader_iphone['compound'] = [sentiments.polarity_scores(i)["compound"] for i in
                                          clean_reviews_vader_iphone["cleaned_review"]]

score = clean_reviews_vader_iphone["compound"].values
sentiment = []
for i in score:
    if i >= 0.05:
        sentiment.append('positive')
    elif i <= -0.05:
        sentiment.append('negative')
    else:
        sentiment.append('neutral')
clean_reviews_vader_iphone["sentiment"] = sentiment

print(clean_reviews_vader_iphone.sentiment.value_counts())
