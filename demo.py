import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils.data_loader import DataLoader
from agent import NewsFeedAgent
from utils.evaluation import RecommendationEvaluator
from models.collaborative_filter import CollaborativeFilter
from models.embedding_model import EmbeddingModel
import os
import time

def main():
    """Main function to run the demo"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='News Feed Agent Demo')
    parser.add_argument('--mode', type=str, default='recommend', choices=['recommend', 'evaluate', 'interactive'],
                       help='Demo mode: recommend, evaluate, or interactive')
    parser.add_argument('--user_id', type=int, default=0,
                       help='User ID for recommendations (default: 0)')
    parser.add_argument('--num_recs', type=int, default=5,
                       help='Number of recommendations to show (default: 5)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data files (default: use built-in data)')
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Load data
    print("Loading data...")
    if args.data_path:
        data_loader = DataLoader(
            use_builtin_data=False,
            news_file=os.path.join(args.data_path, 'news_articles.csv'),
            interactions_file=os.path.join(args.data_path, 'user_interactions.csv')
        )
    else:
        data_loader = DataLoader()
    
    articles_df, interactions_df = data_loader.load_data()
    
    print(f"Loaded {len(articles_df)} articles and {len(interactions_df)} interactions")
    
    # Set the mode
    if args.mode == 'recommend':
        # Simple recommendation mode
        run_recommendation_demo(articles_df, interactions_df, args.user_id, args.num_recs)
    elif args.mode == 'evaluate':
        # Evaluation mode
        run_evaluation_demo(articles_df, interactions_df)
    elif args.mode == 'interactive':
        # Interactive mode
        run_interactive_demo(articles_df, interactions_df)
    else:
        print(f"Unknown mode: {args.mode}")

def run_recommendation_demo(articles_df, interactions_df, user_id, num_recs):
    """Run a simple recommendation demo for a specific user"""
    print(f"Running recommendation demo for user {user_id}...")
    
    # Initialize and train the agent
    agent = NewsFeedAgent()
    agent.train(articles_df, interactions_df)
    
    # Get user profile
    profile = agent.get_user_profile(user_id)
    
    if profile and 'category_preferences' in profile:
        print("\nUser Profile:")
        print(f"Total interactions: {profile['total_interactions']}")
        print("Category preferences:")
        for category, score in sorted(profile['category_preferences'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {category}: {score:.2f}")
        print(f"Average reading time: {profile['avg_reading_time']:.2f} seconds")
    else:
        print(f"No profile data available for user {user_id}")
    
    # Get recommendations
    print(f"\nGenerating recommendations for user {user_id}...")
    recommendations = agent.get_recommendations(user_id, n=num_recs)
    
    if len(recommendations) == 0:
        print("No recommendations available for this user.")
        return
    
    # Display recommendations
    print(f"\nTop {num_recs} Recommendations:")
    for i, (_, rec) in enumerate(recommendations.iterrows()):
        print(f"\n{i+1}. {rec['title']} (Score: {rec['score']:.4f})")
        print(f"   Category: {rec['category']}")
        print(f"   Summary: {rec['summary'][:150]}...")

def run_evaluation_demo(articles_df, interactions_df):
    """Run an evaluation comparing different recommendation approaches"""
    print("Running evaluation demo...")
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator(articles_df, interactions_df)
    
    # Split data into train/test sets
    print("Splitting data into train/test sets...")
    train_df, test_df = evaluator.train_test_split(test_ratio=0.2)
    
    print(f"Training set: {len(train_df)} interactions")
    print(f"Test set: {len(test_df)} interactions")
    
    # Initialize models to compare
    models = [
        ("Collaborative Filtering", NewsFeedAgent(cf_weight=1.0, embedding_weight=0.0)),
        ("Embedding Model", NewsFeedAgent(cf_weight=0.0, embedding_weight=1.0)),
        ("Hybrid Model", NewsFeedAgent(cf_weight=0.5, embedding_weight=0.5))
    ]
    
    # Compare models
    print("Comparing recommendation models...")
    results = evaluator.compare_models(train_df, test_df, models)
    
    # Display results
    print("\nEvaluation Results:")
    print(results.to_string(index=False))
    
    # Plot results
    print("Plotting results...")
    evaluator.plot_results(results)
    print("Plot saved as 'model_comparison.png'")

def run_interactive_demo(articles_df, interactions_df):
    """Run an interactive demo where the user can see recommendations and provide feedback"""
    print("Running interactive demo...")
    
    # Initialize and train the agent
    agent = NewsFeedAgent()
    agent.train(articles_df, interactions_df)
    
    # Get a list of users
    users = sorted(interactions_df['user_id'].unique())
    
    # Select a user
    print("\nAvailable users:")
    for i, user_id in enumerate(users[:10]):
        print(f"{i+1}. User {user_id}")
    
    if len(users) > 10:
        print(f"... and {len(users) - 10} more users")
    
    user_idx = int(input("\nSelect a user (1-10): ")) - 1
    if user_idx < 0 or user_idx >= len(users):
        print("Invalid selection, using user 0")
        user_id = users[0]
    else:
        user_id = users[user_idx]
    
    print(f"\nSelected User {user_id}")
    
    # Show user profile
    profile = agent.get_user_profile(user_id)
    
    if profile and 'category_preferences' in profile:
        print("\nUser Profile:")
        print(f"Total interactions: {profile['total_interactions']}")
        print("Category preferences:")
        for category, score in sorted(profile['category_preferences'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {category}: {score:.2f}")
    
    # Simulation loop
    while True:
        # Get recommendations
        print(f"\nGenerating recommendations for user {user_id}...")
        recommendations = agent.get_recommendations(user_id, n=5)
        
        if len(recommendations) == 0:
            print("No recommendations available for this user.")
            break
        
        # Display recommendations
        print("\nTop 5 Recommendations:")
        for i, (_, rec) in enumerate(recommendations.iterrows()):
            print(f"\n{i+1}. {rec['title']} (Score: {rec['score']:.4f})")
            print(f"   Category: {rec['category']}")
            print(f"   Summary: {rec['summary'][:150]}...")
        
        # Get user feedback
        print("\nOptions:")
        print("1-5: Select an article to read")
        print("v: View user profile")
        print("q: Quit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == 'q':
            break
        elif choice == 'v':
            # Visualize user preferences
            evaluator = RecommendationEvaluator(articles_df, interactions_df)
            evaluator.plot_user_preferences(user_id, agent)
            print(f"User preference chart saved as 'user_{user_id}_preferences.png'")
        elif choice.isdigit() and 1 <= int(choice) <= 5 and int(choice) <= len(recommendations):
            # User selected an article
            article_idx = int(choice) - 1
            selected_article = recommendations.iloc[article_idx]
            
            print(f"\nYou selected: {selected_article['title']}")
            print(f"Category: {selected_article['category']}")
            print(f"\n{selected_article['content'][:500]}...\n")
            
            # Get interaction feedback
            print("How would you rate this article?")
            print("1: Just clicked (low interest)")
            print("2: Liked it")
            print("3: Bookmarked it (high interest)")
            
            rating = input("Enter your rating (1-3): ")
            
            # Process feedback
            if rating.isdigit() and 1 <= int(rating) <= 3:
                rating_map = {1: 'click', 2: 'like', 3: 'bookmark'}
                interaction_type = rating_map[int(rating)]
                
                # Simulate reading time based on interest level
                reading_time = {
                    'click': np.random.randint(10, 60),
                    'like': np.random.randint(60, 180),
                    'bookmark': np.random.randint(180, 300)
                }[interaction_type]
                
                print(f"Recording {interaction_type} interaction with reading time {reading_time} seconds...")
                
                # Update the agent
                agent.update_with_feedback(
                    user_id, 
                    selected_article['article_id'], 
                    interaction_type, 
                    reading_time
                )
                
                print("Feedback recorded! Recommendations will be updated.")
                time.sleep(1)
            else:
                print("Invalid rating. Skipping feedback.")
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()