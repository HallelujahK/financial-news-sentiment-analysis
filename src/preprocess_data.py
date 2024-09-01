import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    A class used to preprocess financial news data for analysis.

    Attributes
    ----------
    data : pd.DataFrame
        The dataset containing financial news.

    Methods
    -------
    handle_missing_values()
        Handles missing values by removing rows with missing data.
    convert_date_column()
        Converts the 'date' column to datetime format.
    clean_text_columns()
        Trims whitespace, replaces underscores with spaces, and normalizes text.
    extract_date_components()
        Extracts year, month, and day from the 'date' column.
    remove_duplicates()
        Removes duplicate rows from the dataset.
    validate_and_clean_urls()
        Validates and cleans the URLs in the 'url' column.
    extract_domain_from_email()
        Extracts the domain name from email addresses in the 'publisher' column.
    preprocess()
        Runs all preprocessing steps in the correct order.
    """

    def __init__(self, filepath):
        """
        Constructs all the necessary attributes for the DataPreprocessor object.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset containing financial news.
        """
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
    
    def handle_missing_values(self):
        """Handles missing values from 'headline' and 'date' by removing rows with any missing data."""
        self.data = self.data.dropna(subset=['headline', 'date'])

    def convert_date_column(self):
        """Converts the 'date' column to datetime format."""
        self.data['date'] = pd.to_datetime(self.data['date'], utc = True, errors='coerce')
        # Removes rows with invalid dates
        self.data = self.data.dropna(subset=['date'])

    def clean_text_columns(self):
        """
        Cleans text columns by trimming whitespace, replacing underscores with spaces,
        and normalizing to lowercase.
        """
        # Trim whitespace and replace underscores in the 'publisher' column
        self.data['publisher'] = self.data['publisher'].str.strip().str.replace('_', ' ')

        # Normalize the 'publisher' and 'headline' columns to lowercase
        self.data['publisher'] = self.data['publisher'].str.lower()
        self.data['headline'] = self.data['headline'].str.lower()

    def extract_date_components(self):
        """
        Extracts year, month, and day from the 'date' column and adds them as new columns.
        """
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day

    def remove_duplicates(self):
        """Removes duplicate rows from the dataset."""
        self.data.drop_duplicates(subset=['headline', 'url', 'date'], inplace=True)

    
    def validate_and_clean_urls(self):
        """Validates and cleans the URLs in the 'url' column."""
        # Ensure URLs are properly formatted and handle any malformed URLs
        self.data['url'] = self.data['url'].apply(lambda x: x if bool(urlparse(x).scheme) else np.nan)

        # Remove rows with invalid URLs
        self.data.dropna(subset=['url'], inplace=True)

    def extract_domain_from_email(self):
        """
        Extracts the domain name from email addresses in the 'publisher' column
        if the publisher is an email address.
        """
        def extract_domain(publisher):
            if '@' in publisher:
                return publisher.split('@')[-1]
            return publisher
        
        self.data['publisher'] = self.data['publisher'].apply(extract_domain)

    def preprocess(self):
        """Runs all preprocessing steps in the correct order."""
        self.handle_missing_values()
        self.convert_date_column()
        self.clean_text_columns()
        self.extract_date_components()
        self.remove_duplicates()
        self.validate_and_clean_urls()
        self.extract_domain_from_email()
        return self.data
