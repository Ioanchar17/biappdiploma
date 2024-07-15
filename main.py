import streamlit as st
import altair as alt
import pandas as pd
import plotly.express as px
from word_count_plots import *
from tpot_exported_pipeline_macbook import *
from text_vader_model_mac import *

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
    tab1, tab2, tab3 = st.tabs(["Reviews Analysis", "Text Analysis", "Sentiment Analysis"])

    ## Review Analysis Tab
    with tab1:
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

    ## Text Analysis Tab
    with tab2:
        option = st.selectbox(
            "Choose N for N-Gram Analysis",
            ("1", "2", "3"), index=None)
        col1, col2, col3 = st.columns(3)
        if option == "1":
            with col1:
                st.write("20 Most Frequent words of Positive Reviews")
                pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 1, 20)
                plot_n_gram(pos_unigram, ["skyblue"])
            with col2:
                st.write("20 Most Frequent words of Neutral Reviews")
                neu_unigram = get_top_n_grams(review_neu['cleaned_review'], 1, 20)
                plot_n_gram(neu_unigram, ["yellow"])
            with col3:
                st.write("20 Most Frequent words of Negative Reviews")
                neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 1, 20)
                plot_n_gram(neu_unigram, ["crimson"])
        elif option == "2":
            with col1:
                st.write("Bigram plot of Positive Reviews")
                pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 2, 20)
                plot_n_gram(pos_unigram, ["skyblue"])
            with col2:
                st.write("Bigram plot of Neutral Reviews")
                neu_unigram = get_top_n_grams(review_neu['cleaned_review'], 2, 20)
                plot_n_gram(neu_unigram, ["yellow"])
            with col3:
                st.write("Bigram plot of Negative Reviews")
                neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 2, 20)
                plot_n_gram(neu_unigram, ["crimson"])
        elif option == "3":
            with col1:
                st.write("Trigram plot of Positive Reviews")
                pos_unigram = get_top_n_grams(review_pos['cleaned_review'], 3, 20)
                plot_n_gram(pos_unigram, ["skyblue"])
            with col2:
                st.write("Trigram plot of Neutral Reviews")
                neu_unigram = get_top_n_grams(review_neu['cleaned_review'], 3, 20)
                plot_n_gram(neu_unigram, ["yellow"])
            with col3:
                st.write("Trigram plot of Negative Reviews")
                neg_unigram = get_top_n_grams(review_neg['cleaned_review'], 3, 20)
                plot_n_gram(neu_unigram, ["crimson"])

    ## Text Analysis Tab
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("My Custom Model (BernoulliNB)")
            visualize_df_sentiment = clean_reviews.groupby(['predicted_sentiment'])[
                'predicted_sentiment'].count().reset_index(name='count')
            visualize_df_sentiment_labels = {0: 'neutral', 1: 'negative', 2: 'positive'}
            visualize_df_sentiment['predicted_sentiment'] = visualize_df_sentiment['predicted_sentiment'].map(visualize_df_sentiment_labels)
            fig = px.pie(visualize_df_sentiment, values='count', names='predicted_sentiment',
                         color='predicted_sentiment',
                         color_discrete_map={'positive': 'skyblue',
                                             'neutral': 'yellow',
                                             'negative': 'crimson'})
            st.plotly_chart(fig)
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

else:
    st.write("Loading App......")
    with st.expander('About', expanded=True):
        st.write('''
            - :red[**LinkedIn Profile**] : https://www.linkedin.com/in/ioannis-charalampidis-4249811a9/
            ''')
