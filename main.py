import streamlit as st
from streamlit_extras.bottom_container import bottom
import altair as alt
import pandas as pd
import plotly.express as px

import clean_dataframes
import official_dataframes
import tpot_exported_pipeline_iphone
import tpot_exported_pipeline_macbook
import tpot_exported_pipeline_airpods
import write_decisions
from word_count_plots import *
from tpot_exported_pipeline_macbook import *
from tpot_exported_pipeline_iphone import *
from tpot_exported_pipeline_airpods import *
from text_vader_model_mac import *
from text_vader_model_iphone import *
from text_vader_model_airpods import *
from official_dataframes import *
from clean_dataframes import *
from write_decisions import *

st.set_page_config(
    page_title="Business Intelligence App for Apple Products",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

with st.sidebar:
    st.title('Business Intelligence App for Apple Products')

    datasets_list = ['iPhone Reviews', 'MacBook Reviews', 'AirPods Reviews']

    selected_dataset = st.selectbox('Select a dataset', datasets_list, index=None)

if selected_dataset == "MacBook Reviews":
    visualize_df = pd.read_csv('Clean_datasets\mac_clean.csv')
    review_pos = visualize_df[visualize_df["rating"] == 'positive'].dropna()
    review_neu = visualize_df[visualize_df["rating"] == 'neutral'].dropna()
    review_neg = visualize_df[visualize_df["rating"] == 'negative'].dropna()

    ## Review Analysis Container
    with st.container():
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
            official_df = official_dataframes.macbook_official_reviews
            fig = px.histogram(official_df['rating'], x="rating")
            st.plotly_chart(fig)

    ## Sentiment Analysis Container
    with st.container():
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

    ## Text Analysis Container
    with st.container():
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

elif selected_dataset == "iPhone Reviews":
    visualize_df = pd.read_csv('Clean_datasets/iphone_clean.csv')
    review_pos = visualize_df[visualize_df["rating"] == 'positive'].dropna()
    review_neu = visualize_df[visualize_df["rating"] == 'neutral'].dropna()
    review_neg = visualize_df[visualize_df["rating"] == 'negative'].dropna()

    ## Review Analysis Container
    with st.container():
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
            official_df = official_dataframes.iphone_official_reviews
            fig = px.histogram(official_df['rating'], x="rating")
            st.plotly_chart(fig)

    ## Sentiment Analysis Container
    with st.container():
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

    ## Text Analysis Container
    with st.container():
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
                # ChatGPT summary of plot
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

elif selected_dataset == "AirPods Reviews":
    visualize_df = pd.read_csv('Clean_datasets/airpods_clean.csv')
    review_pos = visualize_df[visualize_df["rating"] == 'positive'].dropna()
    review_neg = visualize_df[visualize_df["rating"] == 'negative'].dropna()

    ## Review Analysis Container
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header("Ratings of reviews")
            visualize_df = visualize_df.groupby(['rating'])['rating'].count().reset_index(name='count')
            fig = px.pie(visualize_df, values='count', names='rating', color='rating',
                         color_discrete_map={'positive': 'skyblue',
                                             'negative': 'crimson'})
            st.plotly_chart(fig)

    ## Sentiment Analysis Container
    with st.container():
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

    ## Text Analysis Container
    with st.container():
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
    ## Title and apple logo
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.title("Business Intelligence App for Apple Products")
    with col2:
        st.image("images/Apple_logo_white.png", width=55)

    col1, col2, col3 = st.columns(3)
    # iPhone
    with col1:
        st.subheader(f'iPhone 11 Reviews: {len(official_dataframes.iphone_official_reviews)}')
        st.image("images/pngimg.com - iphone_11_PNG38.png", width=200)
        if st.button("Decision for iPhone"):
            st.write(write_decisions.stream_data(write_decisions.iphone_decision))
    # MacBook
    with col2:
        st.subheader(f'MacBook Air M1 Reviews: {len(official_dataframes.macbook_official_reviews)}')
        st.image("images/111883_macbookair.png", width=300)
        if st.button("Decision for MacBook"):
            st.write(write_decisions.stream_data(write_decisions.macbook_decision))
    # AirPods
    with col3:
        st.subheader(f'AirPods 2nd Gen Reviews: {len(official_dataframes.airpods_official_reviews)}')
        st.image("images/apple-airpods.png", width=250)
        if st.button("Decision for AirPods"):
            st.write(write_decisions.stream_data(write_decisions.airpods_decision))

    with bottom():
        with st.expander('About', expanded=False):
            st.write('''
                - LinkedIn Profile : https://www.linkedin.com/in/ioannis-charalampidis-4249811a9/
                ''')


