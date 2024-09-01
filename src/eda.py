import pandas as pd


class DescStat:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        self.data = data

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