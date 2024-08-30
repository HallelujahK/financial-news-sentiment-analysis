import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')

class FinancialNewsEDA:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        # Parse the 'date' column
        self.data['date'] = self.parse_dates(self.data['date'])
        # Extract date components
        self.extract_date_components()


    def parse_dates(self, date_series):
        """Parse the 'date' column with mixed formats."""
        return pd.to_datetime(date_series, errors='coerce', infer_datetime_format=True)

    def extract_date_components(self):
        """Extract date and time components from the 'date' column."""
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day
        self.data['weekday'] = self.data['date'].dt.weekday
        self.data['hour'] = self.data['date'].dt.hour



    def compute_headline_length(self):
        """Compute and summarize basic statistics for headline lengths."""
        self.data['headline_length'] = self.data['headline'].apply(len)
        headline_length_stats = self.data['headline_length'].describe()
        print("Headline Length Statistics:\n", headline_length_stats)
    
    def count_articles_per_publisher(self):
        """Count the number of articles per publisher."""
        publisher_counts = self.data['publisher'].value_counts()
        print("Top 10 Publishers by Article Count:\n", publisher_counts.head(10))
    
    def analyze_publication_dates(self):
        """Analyze the publication dates to see trends over time."""
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day_of_week'] = self.data['date'].dt.day_name()
        
        yearly_trends = self.data['year'].value_counts().sort_index()
        monthly_trends = self.data['month'].value_counts().sort_index()
        weekday_trends = self.data['day_of_week'].value_counts()
        
        print("Yearly Trends:\n", yearly_trends)
        print("Monthly Trends:\n", monthly_trends)
        print("Weekday Trends:\n", weekday_trends)
    
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

    
    
    def analyze_publication_frequency(self):
        """Analyze and plot publication frequency over time."""
        # Set the aesthetics for the plots
        sns.set(style="whitegrid")


        # Drop rows where 'year' or 'month' or 'day' is NaN
        self.data = self.data.dropna(subset=['year', 'month', 'day'])

        # Analyze by year
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.data, x='year', palette='viridis')
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Articles', fontsize=14)
        plt.title('Number of Articles Published per Year', fontsize=16)
        plt.xticks(rotation=45)
        plt.show()

        # Analyze by month
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.data, x='month', palette='viridis')
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Number of Articles', fontsize=14)
        plt.title('Number of Articles Published per Month', fontsize=16)
        plt.xticks(ticks=range(12), labels=[
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ], rotation=45)
        plt.show()

        # Analyze by day
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.data, x='day', palette='viridis')
        plt.xlabel('Day of the Month', fontsize=14)
        plt.ylabel('Number of Articles', fontsize=14)
        plt.title('Number of Articles Published per Day of the Month', fontsize=16)
        plt.show()

        # Analyze by weekday
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.data, x='weekday', palette='viridis')
        plt.xlabel('Weekday', fontsize=14)
        plt.ylabel('Number of Articles', fontsize=14)
        plt.title('Number of Articles Published per Weekday', fontsize=16)
        plt.xticks(ticks=range(7), labels=[
            'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'
        ], rotation=45)
        plt.show()

        # Analyze by hour
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.data, x='hour', palette='viridis')
        plt.xlabel('Hour of the Day', fontsize=14)
        plt.ylabel('Number of Articles', fontsize=14)
        plt.title('Number of Articles Published per Hour of the Day', fontsize=16)
        plt.show()
    
    def analyze_publishers(self):
        """Analyze and visualize publisher data."""
        # Drop rows where 'publisher' is NaN
        self.data = self.data.dropna(subset=['publisher'])

        # Count the number of articles per publisher
        publisher_counts = self.data['publisher'].value_counts()
        top_publishers = publisher_counts.head(10)

        # Plot the top 10 publishers
        plt.figure(figsize=(14, 8))
        sns.barplot(x=top_publishers.index, y=top_publishers.values, palette='viridis')
        plt.xlabel('Publisher', fontsize=14)
        plt.ylabel('Number of Articles', fontsize=14)
        plt.title('Top 10 Publishers by Number of Articles', fontsize=16)
        plt.xticks(rotation=45)
        plt.show()

        # Analyze the type of news reported by each publisher
        # For simplicity, we'll use a basic approach to categorize news types
        self.data['news_type'] = self.data['headline'].apply(self.categorize_news_type)
        publisher_news_type = self.data.groupby('publisher')['news_type'].value_counts().unstack().fillna(0)
        
        print("News Types by Publisher:")
        print(publisher_news_type.head())
        # If email addresses are used, extract and analyze unique domains
        self.data['domain'] = self.data['publisher'].apply(self.extract_domain)
        domain_counts = self.data['domain'].value_counts()
        top_domains = domain_counts.head(10)

        # Plot the top 10 domains
        plt.figure(figsize=(14, 8))
        sns.barplot(x=top_domains.index, y=top_domains.values, palette='viridis')
        plt.xlabel('Domain', fontsize=14)
        plt.ylabel('Number of Publishers', fontsize=14)
        plt.title('Top 10 Domains by Number of Publishers', fontsize=16)
        plt.xticks(rotation=45)
        plt.show()
    
    def categorize_news_type(self, headline):
        """Categorize the news type based on keywords in the headline."""
        if 'FDA approval' in headline:
            return 'FDA Approval'
        elif 'price target' in headline:
            return 'Price Target'
        elif 'earnings' in headline:
            return 'Earnings'
        else:
            return 'Other'

    def extract_domain(self, publisher):
        """Extract domain from an email address or URL."""
        if pd.isna(publisher):
            return 'Unknown'
        match = re.search(r'@([a-zA-Z0-9.-]+)', publisher)
        return match.group(1) if match else 'Unknown'
