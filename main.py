import streamlit as st
import altair as alt
import pandas as pd
import plotly.express as px

import text_vader_model_mac
import tpot_exported_pipeline_iphone
import tpot_exported_pipeline_macbook
import tpot_exported_pipeline_airpods
from word_count_plots import *
from tpot_exported_pipeline_macbook import *
from tpot_exported_pipeline_iphone import *
from tpot_exported_pipeline_airpods import *
from text_vader_model_mac import *
from text_vader_model_iphone import *
from text_vader_model_airpods import *

st.set_page_config(
    page_title="Business Intelligence App from Web Data",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

with st.sidebar:
    st.title('Business Intelligence App from Web Data')

    datasets_list = ['Iphone Reviews', 'Macbook Reviews', 'Airpods Reviews']

    selected_dataset = st.selectbox('Select a dataset', datasets_list, index=None)

if selected_dataset == "Macbook Reviews":
    visualize_df = pd.read_csv('Clean_datasets\mac_clean.csv')
    review_pos = visualize_df[visualize_df["rating"] == 'positive'].dropna()
    review_neu = visualize_df[visualize_df["rating"] == 'neutral'].dropna()
    review_neg = visualize_df[visualize_df["rating"] == 'negative'].dropna()

    ## Review Analysis Tab
    col1, col2 = st.columns(2)
    with col1:
        st.header("Ratings of reviews")
        visualize_df = visualize_df.groupby(['rating'])['rating'].count().reset_index(name='count')
        fig = px.pie(visualize_df, values='count', names='rating', color='rating',
                     color_discrete_map={'positive': 'skyblue',
                                         'neutral': 'yellow',
                                         'negative': 'crimson'})
        st.plotly_chart(fig)

    with col2:
        st.header("Review Rating Distribution")
        official_df = pd.read_csv('Official_datasets/Apple_Macbook_Air_M1_final.csv')
        fig = px.histogram(official_df['rating'], x="rating")
        st.plotly_chart(fig)

    ## Sentiment Analysis Tab

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("My Custom Model (BernoulliNB)")
        visualize_df_sentiment = clean_reviews.groupby(['predicted_sentiment'])[
            'predicted_sentiment'].count().reset_index(name='count')
        visualize_df_sentiment_labels = {0: 'neutral', 1: 'negative', 2: 'positive'}
        visualize_df_sentiment['predicted_sentiment'] = visualize_df_sentiment['predicted_sentiment'].map(
            visualize_df_sentiment_labels)
        fig = px.pie(visualize_df_sentiment, values='count', names='predicted_sentiment',
                     color='predicted_sentiment',
                     color_discrete_map={'positive': 'skyblue',
                                         'neutral': 'yellow',
                                         'negative': 'crimson'})
        st.plotly_chart(fig)
        st.write(f'Accuracy Score : {tpot_exported_pipeline_macbook.acc_score}')
    with col2:
        st.subheader("VADER Model")
        visualize_df_sentiment = clean_reviews_vader.groupby(['sentiment'])[
            'sentiment'].count().reset_index(name='count')
        fig = px.pie(visualize_df_sentiment, values='count', names='sentiment',
                     color='sentiment',
                     color_discrete_map={'positive': 'skyblue',
                                         'neutral': 'yellow',
                                         'negative': 'crimson'})
        st.plotly_chart(fig)

    ## Text Analysis Tab

    option = st.selectbox(
        "Choose N for N-Gram Analysis",
        ("1", "2", "3"), index=None)
    col1, col2, col3 = st.columns(3)
    if option == "1":
        with col1:
            st.subheader("20 Most Frequent words of Positive Reviews")
            pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 1, 20)
            plot_n_gram(pos_unigram, ["skyblue"])
        with col2:
            st.subheader("20 Most Frequent words of Neutral Reviews")
            neu_unigram = get_top_n_grams(review_neu['cleaned_review'], 1, 20)
            plot_n_gram(neu_unigram, ["yellow"])
        with col3:
            st.subheader("20 Most Frequent words of Negative Reviews")
            neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 1, 20)
            plot_n_gram(neu_unigram, ["crimson"])
    elif option == "2":
        with col1:
            st.subheader("Bigram plot of Positive Reviews")
            pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 2, 20)
            plot_n_gram(pos_unigram, ["skyblue"])
        with col2:
            st.subheader("Bigram plot of Neutral Reviews")
            neu_unigram = get_top_n_grams(review_neu['cleaned_review'], 2, 20)
            plot_n_gram(neu_unigram, ["yellow"])
        with col3:
            st.subheader("Bigram plot of Negative Reviews")
            neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 2, 20)
            plot_n_gram(neu_unigram, ["crimson"])
    elif option == "3":
        with col1:
            st.subheader("Trigram plot of Positive Reviews")
            pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 3, 20)
            plot_n_gram(pos_unigram, ["skyblue"])
        with col2:
            st.subheader("Trigram plot of Neutral Reviews")
            neu_unigram = get_top_n_grams(review_neu['cleaned_review'], 3, 20)
            plot_n_gram(neu_unigram, ["yellow"])
        with col3:
            st.subheader("Trigram plot of Negative Reviews")
            neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 3, 20)
            plot_n_gram(neu_unigram, ["crimson"])

elif selected_dataset == "Iphone Reviews":
    visualize_df = pd.read_csv('Clean_datasets/iphone_clean.csv')
    review_pos = visualize_df[visualize_df["rating"] == 'positive'].dropna()
    review_neu = visualize_df[visualize_df["rating"] == 'neutral'].dropna()
    review_neg = visualize_df[visualize_df["rating"] == 'negative'].dropna()

    ## Review Analysis Tab
    col1, col2 = st.columns(2)
    with col1:
        st.header("Ratings of reviews")
        visualize_df = visualize_df.groupby(['rating'])['rating'].count().reset_index(name='count')
        fig = px.pie(visualize_df, values='count', names='rating', color='rating',
                     color_discrete_map={'positive': 'skyblue',
                                         'neutral': 'yellow',
                                         'negative': 'crimson'})
        st.plotly_chart(fig)

    with col2:
        st.header("Review Rating Distribution")
        official_df = pd.read_csv('Official_datasets/Apple_Iphone_11_Reviews_new.CSV', sep=';')
        official_df['rating'] = official_df['rating'].div(10)
        fig = px.histogram(official_df['rating'], x="rating")
        st.plotly_chart(fig)

    ## Sentiment Analysis Tab

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("My Custom Model (LinearSVC)")
        visualize_df_sentiment = clean_reviews_iphone.groupby(['predicted_sentiment'])[
            'predicted_sentiment'].count().reset_index(name='count')
        visualize_df_sentiment_labels = {0: 'neutral', 1: 'negative', 2: 'positive'}
        visualize_df_sentiment['predicted_sentiment'] = visualize_df_sentiment['predicted_sentiment'].map(
            visualize_df_sentiment_labels)
        fig = px.pie(visualize_df_sentiment, values='count', names='predicted_sentiment',
                     color='predicted_sentiment',
                     color_discrete_map={'positive': 'skyblue',
                                         'neutral': 'yellow',
                                         'negative': 'crimson'})
        st.plotly_chart(fig)
        st.write(f'Accuracy Score : {tpot_exported_pipeline_iphone.acc_score}')
    with col2:
        st.subheader("VADER Model")
        visualize_df_sentiment = clean_reviews_vader_iphone.groupby(['sentiment'])[
            'sentiment'].count().reset_index(name='count')
        fig = px.pie(visualize_df_sentiment, values='count', names='sentiment',
                     color='sentiment',
                     color_discrete_map={'positive': 'skyblue',
                                         'neutral': 'yellow',
                                         'negative': 'crimson'})
        st.plotly_chart(fig)
    ## Text Analysis Tab

    option = st.selectbox(
        "Choose N for N-Gram Analysis",
        ("1", "2", "3"), index=None)
    col1, col2, col3 = st.columns(3)
    if option == "1":
        with col1:
            st.subheader("20 Most Frequent words of Positive Reviews")
            pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 1, 20)
            plot_n_gram(pos_unigram, ["skyblue"])
        with col2:
            st.subheader("20 Most Frequent words of Neutral Reviews")
            neu_unigram = get_top_n_grams(review_neu['cleaned_review'], 1, 20)
            plot_n_gram(neu_unigram, ["yellow"])
        with col3:
            st.subheader("20 Most Frequent words of Negative Reviews")
            neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 1, 20)
            plot_n_gram(neu_unigram, ["crimson"])
    elif option == "2":
        with col1:
            st.subheader("Bigram plot of Positive Reviews")
            pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 2, 20)
            plot_n_gram(pos_unigram, ["skyblue"])
        with col2:
            st.subheader("Bigram plot of Neutral Reviews")
            neu_unigram = get_top_n_grams(review_neu['cleaned_review'], 2, 20)
            plot_n_gram(neu_unigram, ["yellow"])
        with col3:
            st.subheader("Bigram plot of Negative Reviews")
            neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 2, 20)
            plot_n_gram(neu_unigram, ["crimson"])
    elif option == "3":
        with col1:
            st.subheader("Trigram plot of Positive Reviews")
            pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 3, 20)
            plot_n_gram(pos_unigram, ["skyblue"])
        with col2:
            st.subheader("Trigram plot of Neutral Reviews")
            neu_unigram = get_top_n_grams(review_neu['cleaned_review'], 3, 20)
            plot_n_gram(neu_unigram, ["yellow"])
        with col3:
            st.subheader("Trigram plot of Negative Reviews")
            neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 3, 20)
            plot_n_gram(neu_unigram, ["crimson"])

elif selected_dataset == "Airpods Reviews":
    visualize_df = pd.read_csv('Clean_datasets/airpods_clean.csv')
    review_pos = visualize_df[visualize_df["rating"] == 'positive'].dropna()
    review_neg = visualize_df[visualize_df["rating"] == 'negative'].dropna()

    ## Review Analysis Tab
    col1, col2 = st.columns(2)
    with col1:
        st.header("Ratings of reviews")
        visualize_df = visualize_df.groupby(['rating'])['rating'].count().reset_index(name='count')
        fig = px.pie(visualize_df, values='count', names='rating', color='rating',
                     color_discrete_map={'positive': 'skyblue',
                                         'negative': 'crimson'})
        st.plotly_chart(fig)
    ## Sentiment Analysis Tab

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("My Custom Model (MultinomialNB)")
        visualize_df_sentiment = clean_reviews_airpods.groupby(['predicted_sentiment'])[
            'predicted_sentiment'].count().reset_index(name='count')
        visualize_df_sentiment_labels = {0: 'negative', 1: 'positive'}
        visualize_df_sentiment['predicted_sentiment'] = visualize_df_sentiment['predicted_sentiment'].map(
            visualize_df_sentiment_labels)
        fig = px.pie(visualize_df_sentiment, values='count', names='predicted_sentiment',
                     color='predicted_sentiment',
                     color_discrete_map={'positive': 'skyblue',
                                         'negative': 'crimson'})
        st.plotly_chart(fig)
        st.write(f'Accuracy Score : {tpot_exported_pipeline_airpods.acc_score}')
    with col2:
        st.subheader("VADER Model")
        visualize_df_sentiment = clean_reviews_vader_airpods.groupby(['sentiment'])[
            'sentiment'].count().reset_index(name='count')
        fig = px.pie(visualize_df_sentiment, values='count', names='sentiment',
                     color='sentiment',
                     color_discrete_map={'positive': 'skyblue',
                                         'negative': 'crimson'})
        st.plotly_chart(fig)
    ## Text Analysis Tab

    option = st.selectbox(
        "Choose N for N-Gram Analysis",
        ("1", "2", "3"), index=None)
    col1, col2 = st.columns(2)
    if option == "1":
        with col1:
            st.subheader("20 Most Frequent words of Positive Reviews")
            pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 1, 20)
            plot_n_gram(pos_unigram, ["skyblue"])
        with col2:
            st.subheader("20 Most Frequent words of Negative Reviews")
            neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 1, 20)
            plot_n_gram(neg_unigram, ["crimson"])
    elif option == "2":
        with col1:
            st.subheader("Bigram plot of Positive Reviews")
            pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 2, 20)
            plot_n_gram(pos_unigram, ["skyblue"])
        with col2:
            st.subheader("Bigram plot of Negative Reviews")
            neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 2, 20)
            plot_n_gram(neg_unigram, ["crimson"])
    elif option == "3":
        with col1:
            st.subheader("Trigram plot of Positive Reviews")
            pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 3, 20)
            plot_n_gram(pos_unigram, ["skyblue"])
        with col2:
            st.subheader("Trigram plot of Negative Reviews")
            neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 3, 20)
            plot_n_gram(neg_unigram, ["crimson"])

else:
    st.write("Loading App......")
    with st.expander('About', expanded=True):
        st.write('''
            - :red[**LinkedIn Profile**] : https://www.linkedin.com/in/ioannis-charalampidis-4249811a9/
            ''')
