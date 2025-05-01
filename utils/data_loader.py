import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import random
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, use_builtin_data=True, news_file=None, interactions_file=None):
        """
        Initialize the data loader
        
        Args:
            use_builtin_data: Whether to use built-in data or load from files
            news_file: Path to news articles CSV (if not using built-in)
            interactions_file: Path to user interactions CSV (if not using built-in)
        """
        self.use_builtin_data = use_builtin_data
        self.news_file = news_file
        self.interactions_file = interactions_file
        
    def load_news_data(self):
        """Load news articles dataset"""
        if self.use_builtin_data:
            # Use the 20 newsgroups dataset as our news articles
            categories = ['rec.sport.baseball', 'sci.space', 'talk.politics.misc', 
                         'comp.graphics', 'sci.med']
            newsgroups = fetch_20newsgroups(subset='train', categories=categories,
                                           remove=('headers', 'footers', 'quotes'))
            
            # Create a date range for the last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            date_range = [start_date + timedelta(days=x) for x in range(31)]
            
            # Create a DataFrame with the news articles
            articles_df = pd.DataFrame({
                'article_id': range(len(newsgroups.data)),
                'title': [f"Article {i}" for i in range(len(newsgroups.data))],
                'content': newsgroups.data,
                'category': [newsgroups.target_names[target] for target in newsgroups.target],
                'date': random.choices(date_range, k=len(newsgroups.data))
            })
            
            # Clean the content (remove very short articles and truncate very long ones)
            articles_df = articles_df[articles_df['content'].str.len() > 100]
            articles_df['summary'] = articles_df['content'].str[:200] + '...'
            
            return articles_df
        else:
            if self.news_file:
                return pd.read_csv(self.news_file)
            else:
                raise ValueError("News file path not provided")
    
    def generate_user_interactions(self, articles_df, num_users=100, interactions_per_user=20):
        """Generate simulated user interaction data"""
        total_articles = len(articles_df)
        
        # Create user profiles with category preferences
        categories = articles_df['category'].unique()
        user_profiles = []
        
        for user_id in range(num_users):
            # Each user has preferences for 1-3 categories
            num_preferred_categories = random.randint(1, 3)
            preferred_categories = random.sample(list(categories), num_preferred_categories)
            
            # Assign preference weights to categories
            category_weights = {}
            for cat in categories:
                if cat in preferred_categories:
                    category_weights[cat] = random.uniform(0.6, 1.0)
                else:
                    category_weights[cat] = random.uniform(0.1, 0.4)
            
            user_profiles.append({
                'user_id': user_id,
                'category_weights': category_weights
            })
        
        # Generate interactions based on user profiles
        interactions = []
        
        for user in user_profiles:
            user_id = user['user_id']
            category_weights = user['category_weights']
            
            # Articles in preferred categories are more likely to be selected
            article_weights = []
            for _, article in articles_df.iterrows():
                article_weights.append(category_weights[article['category']])
            
            # Normalize weights
            article_weights = np.array(article_weights) / sum(article_weights)
            
            # Sample articles based on weights
            interacted_articles = np.random.choice(
                articles_df['article_id'].values, 
                size=min(interactions_per_user, total_articles),
                replace=False,
                p=article_weights
            )
            
            # Generate interactions
            for article_id in interacted_articles:
                # Generate interaction type (click, like, bookmark)
                interaction_type = random.choices(
                    ['click', 'like', 'bookmark'],
                    weights=[0.7, 0.2, 0.1],
                    k=1
                )[0]
                
                # Generate reading time (in seconds)
                if interaction_type == 'click':
                    reading_time = random.randint(5, 300)
                else:
                    reading_time = random.randint(30, 600)
                
                interactions.append({
                    'user_id': user_id,
                    'article_id': article_id,
                    'interaction_type': interaction_type,
                    'reading_time': reading_time,
                    'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
                })
        
        return pd.DataFrame(interactions)
    
    def load_user_interactions(self, articles_df=None):
        """Load user interaction data"""
        if self.use_builtin_data:
            if articles_df is None:
                articles_df = self.load_news_data()
            return self.generate_user_interactions(articles_df)
        else:
            if self.interactions_file:
                return pd.read_csv(self.interactions_file)
            else:
                raise ValueError("Interactions file path not provided")
    
    def load_data(self):
        """Load both news and interaction data"""
        articles_df = self.load_news_data()
        interactions_df = self.load_user_interactions(articles_df)
        
        return articles_df, interactions_df