import pandas as pd
import numpy as np
from models.collaborative_filter import CollaborativeFilter
from models.embedding_model import EmbeddingModel

class NewsFeedAgent:
    """
    Personalized news feed agent that combines collaborative filtering
    and embedding-based recommendations
    """
    
    def __init__(self, cf_weight=0.4, embedding_weight=0.6):
        """
        Initialize the news feed agent
        
        Args:
            cf_weight: Weight given to collaborative filtering recommendations
            embedding_weight: Weight given to embedding-based recommendations
        """
        self.collaborative_filter = CollaborativeFilter(method='user')
        self.embedding_model = EmbeddingModel(embedding_size=100)
        self.cf_weight = cf_weight
        self.embedding_weight = embedding_weight
        self.articles_df = None
        self.interactions_df = None
        
    def train(self, articles_df, interactions_df):
        """
        Train the recommendation models
        
        Args:
            articles_df: DataFrame with article data
            interactions_df: DataFrame with user interaction data
        """
        print("Training models...")
        self.articles_df = articles_df
        self.interactions_df = interactions_df
        
        # Train collaborative filtering model
        print("Training collaborative filtering model...")
        self.collaborative_filter.fit(interactions_df, articles_df)
        
        # Train embedding model
        print("Training embedding model...")
        self.embedding_model.fit(articles_df, interactions_df)
        
        # Fine-tune embedding model based on user interactions
        print("Fine-tuning embedding model...")
        self.embedding_model.fine_tune(interactions_df)
        
        print("Training complete!")
    
    def get_recommendations(self, user_id, n=10, exclude_seen=True):
        """
        Generate personalized recommendations for a user
        
        Args:
            user_id: ID of the user to recommend articles to
            n: Number of recommendations to generate
            exclude_seen: Whether to exclude articles the user has already interacted with
            
        Returns:
            DataFrame with recommended articles including ranking information
        """
        # Get recommendations from both models
        cf_recs = self.collaborative_filter.recommend(user_id, n=n*2, exclude_seen=exclude_seen)
        emb_recs = self.embedding_model.recommend(
            user_id, self.articles_df, n=n*2, exclude_seen=exclude_seen, interactions_df=self.interactions_df
        )
        
        # Create a unified ranking of recommendations
        all_recs = {}
        
        # Add collaborative filtering recommendations with their score
        for i, article_id in enumerate(cf_recs):
            all_recs[article_id] = {
                'cf_score': 1.0 - (i / (2 * n)),  # Normalize score between 0-1
                'emb_score': 0.0  # Default score for embedding model
            }
        
        # Add embedding model recommendations with their score
        for i, article_id in enumerate(emb_recs):
            if article_id in all_recs:
                all_recs[article_id]['emb_score'] = 1.0 - (i / (2 * n))
            else:
                all_recs[article_id] = {
                    'cf_score': 0.0,  # Default score for collaborative filtering
                    'emb_score': 1.0 - (i / (2 * n))
                }
        
        # Calculate combined score with weights
        for article_id in all_recs:
            all_recs[article_id]['combined_score'] = (
                self.cf_weight * all_recs[article_id]['cf_score'] +
                self.embedding_weight * all_recs[article_id]['emb_score']
            )
        
        # Sort by combined score and take top N
        top_recs = sorted(
            all_recs.items(), 
            key=lambda x: x[1]['combined_score'], 
            reverse=True
        )[:n]
        
        # Create recommendations DataFrame with article details
        rec_ids = [article_id for article_id, _ in top_recs]
        rec_scores = [scores['combined_score'] for _, scores in top_recs]
        
        # Get article details
        rec_articles = self.articles_df[self.articles_df['article_id'].isin(rec_ids)].copy()
        
        # Add ranking information
        id_to_rank = {article_id: i+1 for i, article_id in enumerate(rec_ids)}
        id_to_score = {article_id: score for article_id, score in zip(rec_ids, rec_scores)}
        
        rec_articles['rank'] = rec_articles['article_id'].map(id_to_rank)
        rec_articles['score'] = rec_articles['article_id'].map(id_to_score)
        
        # Sort by rank
        rec_articles = rec_articles.sort_values('rank')
        
        return rec_articles
    
    def update_with_feedback(self, user_id, article_id, interaction_type, reading_time):
        """
        Update recommendations based on user feedback
        
        Args:
            user_id: ID of the user
            article_id: ID of the article interacted with
            interaction_type: Type of interaction (click, like, bookmark)
            reading_time: Time spent reading the article (in seconds)
        """
        # Create a new interaction record
        new_interaction = pd.DataFrame({
            'user_id': [user_id],
            'article_id': [article_id],
            'interaction_type': [interaction_type],
            'reading_time': [reading_time],
            'timestamp': [pd.Timestamp.now().isoformat()]
        })
        
        # Update the interaction data
        self.interactions_df = pd.concat([self.interactions_df, new_interaction], ignore_index=True)
        
        # Update user embedding
        self.embedding_model.update_user_embedding(user_id, new_interaction)
        
        # For significant changes (likes/bookmarks), we might want to fine-tune the models
        if interaction_type in ['like', 'bookmark']:
            # Fine-tune embedding model with just this interaction
            self.embedding_model.fine_tune(new_interaction, epochs=1)
            
            # For a production system, we would periodically retrain the collaborative filter
            # rather than after every interaction, as it's more computationally intensive
        
    def get_user_profile(self, user_id):
        """
        Get a summary of the user's profile and preferences
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with user profile information
        """
        if self.interactions_df is None:
            return None
            
        # Get user interactions
        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        
        if user_interactions.empty:
            return {'user_id': user_id, 'message': 'No interaction data available for this user.'}
            
        # Get interacted articles
        interacted_articles = self.articles_df[
            self.articles_df['article_id'].isin(user_interactions['article_id'])
        ]
        
        # Calculate category preferences
        category_counts = interacted_articles['category'].value_counts()
        total_interactions = len(interacted_articles)
        category_prefs = {
            category: count / total_interactions 
            for category, count in category_counts.items()
        }
        
        # Calculate interaction statistics
        interaction_counts = user_interactions['interaction_type'].value_counts().to_dict()
        avg_reading_time = user_interactions['reading_time'].mean()
        
        # Create profile summary
        profile = {
            'user_id': user_id,
            'total_interactions': len(user_interactions),
            'category_preferences': category_prefs,
            'interaction_stats': interaction_counts,
            'avg_reading_time': avg_reading_time,
            'most_recent_interaction': user_interactions['timestamp'].max()
        }
        
        return profile