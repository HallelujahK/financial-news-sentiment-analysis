import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

nltk.download('punkt')

class TextAnalyzer:
    def __init__(self, df):
          """Initialize the TextAnalyzer with the DataFrame containing the data"""
          self.df = df
    def perform_sentiment_analysis(self):
        """Perform sentiment analysis on headlines."""
        self.data['sentiment'] = self.data['headline'].apply(self.get_sentiment)
        sentiment_distribution = self.data['sentiment'].value_counts()
        print("Sentiment Distribution:\n", sentiment_distribution)
    
    @staticmethod
    def get_sentiment(text):
        """Helper method to determine the sentiment of a headline."""
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        if sentiment_score > 0:
            return 'positive'
        elif sentiment_score < 0:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_keywords(self, max_features=10):
        """Extract common keywords from the headlines."""
        tokenized_headlines = self.data['headline'].apply(nltk.word_tokenize)
        vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
        X = vectorizer.fit_transform(self.data['headline'])
        
        keywords = vectorizer.get_feature_names_out()
        print("Top keywords:", keywords)
