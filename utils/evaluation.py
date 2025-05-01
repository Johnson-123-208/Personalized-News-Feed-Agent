import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationEvaluator:
    """
    Evaluate recommendation performance using various metrics
    """
    
    def __init__(self, articles_df, interactions_df):
        """
        Initialize the evaluator
        
        Args:
            articles_df: DataFrame with article data
            interactions_df: DataFrame with user interaction data
        """
        self.articles_df = articles_df
        self.interactions_df = interactions_df
    
    def train_test_split(self, test_ratio=0.2, chronological=True):
        """
        Split interaction data into training and test sets
        
        Args:
            test_ratio: Proportion of data to use for testing
            chronological: Whether to split chronologically or randomly
            
        Returns:
            train_df, test_df: Training and test DataFrames
        """
        if chronological:
            # Sort by timestamp
            sorted_df = self.interactions_df.sort_values('timestamp')
            
            # Split based on time
            split_idx = int(len(sorted_df) * (1 - test_ratio))
            train_df = sorted_df.iloc[:split_idx].copy()
            test_df = sorted_df.iloc[split_idx:].copy()
        else:
            # Random split
            train_df = self.interactions_df.sample(frac=(1-test_ratio), random_state=42)
            test_df = self.interactions_df.drop(train_df.index)
        
        return train_df, test_df
    
    def get_ground_truth(self, test_df, like_only=True):
        """
        Create ground truth sets for evaluation
        
        Args:
            test_df: Test DataFrame
            like_only: If True, only consider likes and bookmarks as positive interactions
            
        Returns:
            Dictionary mapping user IDs to sets of relevant article IDs
        """
        ground_truth = {}
        
        # Group by user
        user_groups = test_df.groupby('user_id')
        
        for user_id, group in user_groups:
            if like_only:
                # Only consider likes and bookmarks as positive interactions
                relevant_items = set(group[
                    (group['interaction_type'] == 'like') | 
                    (group['interaction_type'] == 'bookmark')
                ]['article_id'].values)
            else:
                # Consider all interactions as positive
                relevant_items = set(group['article_id'].values)
            
            if relevant_items:  # Only include users with at least one relevant item
                ground_truth[user_id] = relevant_items
        
        return ground_truth
    
    def evaluate_recommendations(self, agent, ground_truth, k=10):
        """
        Evaluate recommendations for all users in ground truth
        
        Args:
            agent: Trained NewsFeedAgent
            ground_truth: Dictionary mapping user IDs to sets of relevant article IDs
            k: Number of recommendations to consider
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Initialize metrics
        precisions = []
        recalls = []
        f1_scores = []
        ndcg_scores = []
        
        # Evaluate for each user
        for user_id, relevant_items in ground_truth.items():
            # Get recommendations
            recommendations = agent.get_recommendations(user_id, n=k, exclude_seen=False)
            rec_items = set(recommendations['article_id'].values)
            
            # Calculate precision
            precision = len(rec_items.intersection(relevant_items)) / len(rec_items) if rec_items else 0
            precisions.append(precision)
            
            # Calculate recall
            recall = len(rec_items.intersection(relevant_items)) / len(relevant_items) if relevant_items else 0
            recalls.append(recall)
            
            # Calculate F1 score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
            
            # Calculate NDCG
            # Create binary relevance array for recommendations
            if len(recommendations) > 0:
                y_true = np.zeros(len(recommendations))
                for i, article_id in enumerate(recommendations['article_id'].values):
                    if article_id in relevant_items:
                        y_true[i] = 1
                
                # Create predicted scores array
                y_score = recommendations['score'].values
                
                # Calculate NDCG
                ndcg = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1))
                ndcg_scores.append(ndcg)
        
        # Calculate average metrics
        avg_metrics = {
            'precision@k': np.mean(precisions),
            'recall@k': np.mean(recalls),
            'f1@k': np.mean(f1_scores),
            'ndcg@k': np.mean(ndcg_scores) if ndcg_scores else 0,
            'num_users': len(ground_truth)
        }
        
        return avg_metrics
    
    def compare_models(self, train_df, test_df, models_to_compare):
        """
        Compare different recommendation models
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            models_to_compare: List of (name, model) tuples
            
        Returns:
            DataFrame with comparative metrics
        """
        # Get ground truth
        ground_truth = self.get_ground_truth(test_df)
        
        # Evaluate each model
        results = []
        
        for model_name, model in models_to_compare:
            # Train the model
            model.train(self.articles_df, train_df)
            
            # Evaluate
            metrics = self.evaluate_recommendations(model, ground_truth)
            metrics['model'] = model_name
            
            results.append(metrics)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def plot_results(self, results_df):
        """
        Plot comparison results
        
        Args:
            results_df: DataFrame with comparison results
        """
        # Set plot style
        plt.style.use('seaborn-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get metrics to plot
        metrics = ['precision@k', 'recall@k', 'f1@k', 'ndcg@k']
        
        # Create bar positions
        bar_width = 0.2
        r = np.arange(len(metrics))
        
        # Plot bars for each model
        models = results_df['model'].values
        for i, model in enumerate(models):
            model_results = results_df[results_df['model'] == model]
            values = [model_results[metric].values[0] for metric in metrics]
            ax.bar(r + i * bar_width, values, width=bar_width, label=model)
        
        # Add labels and legend
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Recommendation Model Comparison')
        ax.set_xticks(r + bar_width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
    
    def plot_user_preferences(self, user_id, agent):
        """
        Plot user category preferences
        
        Args:
            user_id: ID of the user to visualize
            agent: Trained NewsFeedAgent
        """
        # Get user profile
        profile = agent.get_user_profile(user_id)
        
        if profile is None or 'category_preferences' not in profile:
            print(f"No profile data available for user {user_id}")
            return
        
        # Plot category preferences
        plt.figure(figsize=(10, 6))
        categories = list(profile['category_preferences'].keys())
        values = list(profile['category_preferences'].values())
        
        # Sort by preference value
        sorted_indices = np.argsort(values)[::-1]
        categories = [categories[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        plt.bar(categories, values)
        plt.title(f"Category Preferences for User {user_id}")
        plt.xlabel("Category")
        plt.ylabel("Preference Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'user_{user_id}_preferences.png')
        plt.close()