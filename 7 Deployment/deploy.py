import streamlit as st
import joblib
import wordninja
import contractions
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import plotly.express as px

lr_model = joblib.load('best_lr_model.pkl')
lr_vectorizer = joblib.load('lr_tfidf_vectorizer.pkl')
bertopic_model = joblib.load('berTopic.pkl')

nltk.download('stopwords')
nltk.download('wordnet')

def lowercase_text(text):
    return text.lower()

def replace_apostrophe(text):
    return text.replace("’", "'")

def expand_contractions(text):
    return contractions.fix(text)

def convert_emojis_with_text(text):
    return emoji.demojize(text, language="en").replace('_', ' ').replace(':', ' ')

def remove_cnn(text):
    return text.replace('cnn', '')

def remove_short_words(text):
    return ' '.join([word for word in text.split() if len(word) > 2])

def remove_symbols_digits(text):
    return re.sub('[^a-zA-Z\s]', ' ', text)

def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def remove_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)

def remove_whitespace(text):
    return ' '.join(text.split())

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([token for token in text.split() if token.lower() not in stop_words])

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(token) for token in text.split()])

def preprocess_text(text):
    text = lowercase_text(text)
    text = replace_apostrophe(text)
    text = ' '.join(wordninja.split(text))
    text = expand_contractions(text)
    text = convert_emojis_with_text(text)
    text = remove_cnn(text)
    text = remove_short_words(text)
    text = remove_symbols_digits(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_whitespace(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

st.title("Sentiment Analysis and Topic Modelling of News")

news_headline = st.text_input("Enter News Headline (For Sentiment Analysis):")

if news_headline:
    preprocessed_headline = preprocess_text(news_headline)

    headline_vector = lr_vectorizer.transform([preprocessed_headline])
    sentiment_prediction = lr_model.predict(headline_vector)[0]

    if sentiment_prediction == 1:
        st.write("Sentiment: Positive")
    elif sentiment_prediction == 0:
        st.write("Sentiment: Negative")
    else:
        st.write("Sentiment: Neutral")

news_article = st.text_area("Enter News Article Content (For Topic Modelling):")

if news_article:
    preprocessed_article = preprocess_text(news_article)

    topics, _ = bertopic_model.transform([preprocessed_article])

    identified_topic = topics[0]
    st.write(f"Identified Topic: {identified_topic}")

    top_words = bertopic_model.get_topic(identified_topic)

    if top_words:
        st.write("Top 10 Words for the Identified Topic:")
        for i, (word, _) in enumerate(top_words[:10]):
            st.write(f"{i + 1}. {word}")
    else:
        st.write("No top words available for this topic.")

    st.header("Correlation Analysis")

    file_path = 'cnn_news_articles_final_cleaned.csv'
    df = pd.read_csv(file_path)

    df['date published'] = pd.to_datetime(df['date published'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')

    df['year_month'] = df['date published'].dt.to_period('M')

    monthly_avg_sentiment = df.groupby(['year_month', 'category'])['sentiment_score'].mean().reset_index()

    pivot_data = monthly_avg_sentiment.pivot(index='year_month', columns='category', values='sentiment_score')

    pivot_data.reset_index(inplace=True)
    pivot_data['year_month'] = pivot_data['year_month'].astype(str)

    melted_data = pivot_data.melt(id_vars='year_month', var_name='category', value_name='sentiment_score')

    fig1 = px.line(
        melted_data, 
        x='year_month', 
        y='sentiment_score', 
        color='category',
        title='Average Sentiment Scores over time across All Categories',
        labels={
            'year_month': 'Date (Year-Month)',
            'sentiment_score': 'Sentiment Score',
            'category': 'Category'
        }
    )
    fig1.update_layout(
        xaxis_title='Date (Year-Month)',
        yaxis_title='Sentiment Score',
        legend_title='Category',
        xaxis=dict(tickangle=45),
        width=1000,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig1)

    df_sampled = df.sample(n=9570, random_state=42)

    df_sampled['date published'] = pd.to_datetime(df_sampled['date published'])  

    timestamps = df_sampled['date published'].tolist()
    texts = df_sampled['text'].tolist()

    topics_over_time = bertopic_model.topics_over_time(docs=texts, 
                                                        timestamps=timestamps, 
                                                        nr_bins=20)

    filtered_topics_over_time = topics_over_time[topics_over_time['Topic'] == identified_topic]

    fig2 = bertopic_model.visualize_topics_over_time(filtered_topics_over_time)

    st.plotly_chart(fig2)