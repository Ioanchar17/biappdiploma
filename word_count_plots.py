import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import streamlit as st


def get_top_n_grams(corpus, n_gram_value, n=None):
    vector = CountVectorizer(ngram_range=(n_gram_value, n_gram_value)).fit(corpus)

    # We will used bag of words representation
    bow = vector.transform(corpus)
    sum_words = bow.sum(axis=0)

    # Determine frequency for the chart
    words_freq = [(word, sum_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:n]


def plot_n_gram(common_words, color):
    df = pd.DataFrame(common_words, columns=['ReviewText', 'count'])
    fig = px.histogram(df, x="ReviewText", y="count", color_discrete_sequence=color)
    st.plotly_chart(fig, use_container_width=True)
