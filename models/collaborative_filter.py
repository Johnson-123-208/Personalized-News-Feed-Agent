import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class CollaborativeFilter:
    """
    Collaborative filtering recommendation system that can use both 
    user-based and item-based approaches
    """
    
    def __init__(self, method='user'):
        """
        Initialize the collaborative filter
        
        Args:
            method: 'user' for user-based or 'item' for item-based collaborative filtering
        """
        self.method = method
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.articles_df = None
        self.users = None
        self.items = None
        
    def fit(self, interactions_df, articles_df):
        """
        Build the user-item matrix and compute similarity matrix
        
        Args:
            interactions_df: DataFrame with user interaction data
            articles_df: DataFrame with article data
        """
        self.articles_df = articles_df
        
        # Create user-item interaction matrix
        # We'll use reading time as the interaction strength
        # For likes and bookmarks, we'll boost the interaction strength
        
        # Apply weights to different interaction types
        interactions_df['weight'] = interactions_df['reading_time']
        interactions_df.loc[interactions_df['interaction_type'] == 'like', 'weight'] *= 2
        interactions_df.loc[interactions_df['interaction_type'] == 'bookmark', 'weight'] *= 3
        
        # In case of multiple interactions between user and item, take the maximum weight
        grouped_interactions = interactions_df.groupby(['user_id', 'article_id'])['weight'].max().reset_index()
        
        # Create sparse matrix
        self.users = sorted(grouped_interactions['user_id'].unique())
        self.items = sorted(articles_df['article_id'].unique())
        
        user_map = {user: i for i, user in enumerate(self.users)}
        item_map = {item: i for i, item in enumerate(self.items)}
        
        user_indices = [user_map[user] for user in grouped_interactions['user_id']]
        item_indices = [item_map[item] for item in grouped_interactions['article_id']]
        
        # Create sparse matrix with interaction weights
        self.user_item_matrix = csr_matrix((grouped_interactions['weight'], 
                                           (user_indices, item_indices)),
                                          shape=(len(self.users), len(self.items)))
        
        # Compute similarity matrix based on method
        if self.method == 'user':
            # User-based: calculate similarity between users
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        else:
            # Item-based: calculate similarity between items
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            
    def recommend(self, user_id, n=10, exclude_seen=True):
        """
        Generate recommendations for a user
        
        Args:
            user_id: ID of the user to recommend articles to
            n: Number of recommendations to generate
            exclude_seen: Whether to exclude articles the user has already interacted with
            
        Returns:
            List of recommended article IDs
        """
        if user_id not in self.users:
            # If this is a new user, return most popular articles
            return self._recommend_popular(n)
        
        user_idx = self.users.index(user_id)
        
        if self.method == 'user':
            # User-based recommendation
            user_vector = self.user_item_matrix[user_idx].toarray().flatten()
            
            # Find similar users (excluding self)
            similar_users = self.similarity_matrix[user_idx].argsort()[::-1][1:]
            
            # Get top similar users
            top_similar_users = similar_users[:30]  # Consider top 30 similar users
            
            # Calculate weighted ratings for all items
            weighted_ratings = np.zeros(len(self.items))
            similarity_sums = np.zeros(len(self.items))
            
            for similar_user_idx in top_similar_users:
                # Get similarity score
                similarity = self.similarity_matrix[user_idx, similar_user_idx]
                
                # Skip users with low similarity
                if similarity <= 0:
                    continue
                
                # Get this user's ratings
                similar_user_vector = self.user_item_matrix[similar_user_idx].toarray().flatten()
                
                # Add weighted ratings only for items the similar user has interacted with
                for item_idx in range(len(self.items)):
                    if similar_user_vector[item_idx] > 0:
                        weighted_ratings[item_idx] += similarity * similar_user_vector[item_idx]
                        similarity_sums[item_idx] += similarity
            
            # Normalize ratings
            for i in range(len(weighted_ratings)):
                if similarity_sums[i] > 0:
                    weighted_ratings[i] /= similarity_sums[i]
            
            # Create candidates
            if exclude_seen:
                for item_idx in range(len(self.items)):
                    if user_vector[item_idx] > 0:
                        weighted_ratings[item_idx] = -1
            
            # Get top N recommendations
            top_items_idx = weighted_ratings.argsort()[::-1][:n]
            
        else:  # Item-based
            user_vector = self.user_item_matrix[user_idx].toarray().flatten()
            
            # Initialize predictions
            predictions = np.zeros(len(self.items))
            
            # For each item
            for item_idx in range(len(self.items)):
                # Skip if user has already interacted with this item and we want to exclude seen items
                if exclude_seen and user_vector[item_idx] > 0:
                    continue
                
                # Find similar items the user has interacted with
                weighted_sum = 0
                similarity_sum = 0
                
                for interacted_item_idx in np.where(user_vector > 0)[0]:
                    similarity = self.similarity_matrix[item_idx, interacted_item_idx]
                    
                    # Only consider positively correlated items
                    if similarity > 0:
                        weighted_sum += similarity * user_vector[interacted_item_idx]
                        similarity_sum += similarity
                
                # Normalize prediction
                if similarity_sum > 0:
                    predictions[item_idx] = weighted_sum / similarity_sum
            
            # Get top N recommendations
            top_items_idx = predictions.argsort()[::-1][:n]
        
        # Convert indices to article IDs
        recommendations = [self.items[idx] for idx in top_items_idx]
        
        return recommendations
    
    def _recommend_popular(self, n=10):
        """Recommend most popular articles for new users"""
        if self.user_item_matrix is None:
            return []
        
        # Sum interactions for each item
        item_popularity = self.user_item_matrix.sum(axis=0).A1
        
        # Get top N popular items
        popular_items_idx = item_popularity.argsort()[::-1][:n]
        popular_recommendations = [self.items[idx] for idx in popular_items_idx]
        
        return popular_recommendations