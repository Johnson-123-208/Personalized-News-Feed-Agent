import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
from PIL import Image

# Add the project root to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your project modules
try:
    from agent import NewsFeedAgent  # Fixed class name from NewsRecommendationAgent to NewsFeedAgent
    from utils.data_loader import DataLoader
except ImportError:
    st.error("Failed to import required modules. Make sure your project structure is correct.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Personalized News Feed Agent",
    page_icon="📰",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 42px !important;
        font-weight: bold;
        color: #5E35B1;
        margin-bottom: 2rem;
    }
    .subtitle {
        font-size: 26px !important;
        color: #5E35B1;
        margin-bottom: 1rem;
    }
    .category-tag {
        background-color: #E0D4FF;
        color: #5E35B1;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .article-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #5E35B1;
    }
    .score-badge {
        background-color: #5E35B1;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        float: right;
    }
    .article-title {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 0.5rem;
    }
    .article-summary {
        font-size: 14px;
        color: #555;
    }
    .loading {
        text-align: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'users' not in st.session_state:
    st.session_state.users = []
if 'articles' not in st.session_state:
    st.session_state.articles = pd.DataFrame()
if 'interactions' not in st.session_state:
    st.session_state.interactions = pd.DataFrame()

# Title
st.markdown('<p class="title">Personalized News Feed Agent</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("PurpleMerit.png", width=200)
    st.markdown("## Control Panel")
    
    if not st.session_state.data_loaded:
        if st.button("Load Data & Train Models"):
            with st.spinner("Loading data and training models..."):
                # Load data
                data_loader = DataLoader()
                st.session_state.articles, st.session_state.interactions = data_loader.load_data()
                
                # Initialize and train agent
                st.session_state.agent = NewsFeedAgent()  
                st.session_state.agent.train(st.session_state.articles, st.session_state.interactions)
                
                # Get unique users
                st.session_state.users = sorted(st.session_state.interactions['user_id'].unique())
                st.session_state.data_loaded = True
    
    if st.session_state.data_loaded:
        st.success("✅ Models trained and ready!")
        
        # User selection
        selected_user = st.selectbox(
            "Select User ID", 
            options=st.session_state.users,
            index=0
        )
        
        num_recommendations = st.slider(
            "Number of Recommendations", 
            min_value=3, 
            max_value=10, 
            value=5
        )
        
        # Category filters
        if len(st.session_state.articles) > 0:
            categories = ['All'] + sorted(st.session_state.articles['category'].unique().tolist())
            selected_categories = st.multiselect(
                "Filter by Category",
                options=categories[1:],  # Exclude 'All'
                default=[]
            )
        
        # Recommendation method
        rec_method = st.radio(
            "Recommendation Method",
            options=["Hybrid", "Collaborative Filtering", "Content-Based"],
            index=0
        )
        
        # Generate recommendations button
        generate_btn = st.button("Generate Recommendations")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This application demonstrates a personalized news recommendation system using collaborative filtering and fine-tuned embeddings.")
    st.markdown("Created for PurpleMerit AI/ML Engineer Position")

# Main content area
if not st.session_state.data_loaded:
    st.info("👈 Please load data and train models from the sidebar to get started.")
    
    # Project description
    st.markdown("## Project Overview")
    st.markdown("""
    This Personalized News Feed Agent combines collaborative filtering and fine-tuned embeddings to deliver tailored news recommendations to users based on their reading preferences and behaviors.
    
    ### Key Features:
    
    * **Hybrid Recommendation Engine**: Combines collaborative filtering with content-based approaches
    * **User Profile Modeling**: Builds detailed user profiles based on interaction history
    * **Adaptive Learning**: Updates recommendations based on user feedback
    * **Category-Based Filtering**: Allows filtering recommendations by topic categories
    
    ### Technologies Used:
    
    * **Collaborative Filtering**: Identifies similar users and their content preferences
    * **Doc2Vec Embeddings**: Creates semantic representations of articles
    * **Fine-tuned Models**: Adapts pre-trained models to news domain
    * **Streamlit**: Powers this interactive web interface
    """)
    
    # Sample visualization
    st.markdown("## Sample Visualization")
    
    # Create a sample visualization
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Sample data for visualization
    categories = ['Politics', 'Sports', 'Technology', 'Entertainment', 'Business']
    counts = [35, 25, 20, 15, 5]
    
    # First subplot - Bar chart of categories
    ax[0].bar(categories, counts, color='#5E35B1')
    ax[0].set_title('Category Distribution in Recommendations')
    ax[0].set_ylabel('Count')
    ax[0].set_xlabel('Category')
    
    # Second subplot - Pie chart
    ax[1].pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(categories)))
    ax[1].set_title('Proportion of Categories')
    
    plt.tight_layout()
    st.pyplot(fig)

elif generate_btn:
    # Get user profile
    user_profile = st.session_state.agent.get_user_profile(selected_user)
    
    # Display user profile
    st.markdown('<p class="subtitle">User Profile</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**User ID:** {selected_user}")
        st.markdown(f"**Total interactions:** {user_profile.get('total_interactions', 0)}")  # Using get() for safer access
        
        if 'avg_reading_time' in user_profile:  # Changed to match actual field in NewsFeedAgent
            st.markdown(f"**Average reading time:** {user_profile['avg_reading_time']:.2f} seconds")
    
    with col2:
        if 'category_preferences' in user_profile:
            st.markdown("**Category preferences:**")
            for category, weight in sorted(user_profile['category_preferences'].items(), key=lambda x: x[1], reverse=True):
                st.markdown(f"- {category}: {weight:.2f}")
    
    # Create a visualization of user preferences
    if 'category_preferences' in user_profile and user_profile['category_preferences']:
        st.markdown("### Category Preference Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        categories = list(user_profile['category_preferences'].keys())
        weights = list(user_profile['category_preferences'].values())
        
        # Sort by weight
        sorted_data = sorted(zip(categories, weights), key=lambda x: x[1], reverse=True)
        categories = [item[0] for item in sorted_data]
        weights = [item[1] for item in sorted_data]
        
        sns.barplot(x=categories, y=weights, palette="viridis", ax=ax)
        ax.set_title('User Category Preferences')
        ax.set_ylabel('Preference Weight')
        ax.set_xlabel('Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Generate recommendations based on method
    with st.spinner("Generating personalized recommendations..."):
        # Adapting method parameters to match NewsFeedAgent implementation
        method_param = {"Hybrid": None, "Collaborative Filtering": "cf_only", "Content-Based": "emb_only"}
        method = method_param.get(rec_method)
        
        recommendations = st.session_state.agent.get_recommendations(
            selected_user, 
            n=num_recommendations,
            exclude_seen=True  # Added parameter to match NewsFeedAgent implementation
        )
    
    # Filter by selected categories if any
    if selected_categories:
        recommendations = recommendations[recommendations['category'].isin(selected_categories)]
    
    # Display recommendations
    st.markdown('<p class="subtitle">Top Recommendations</p>', unsafe_allow_html=True)
    
    if len(recommendations) == 0:
        st.warning("No recommendations found matching the selected criteria.")
    else:
        for i, (_, rec) in enumerate(recommendations.iterrows()):
            st.markdown(
                f"""
                <div class="article-card">
                    <span class="score-badge">Score: {rec.get('score', 0):.4f}</span>
                    <p class="article-title">
                        {i+1}. {rec.get('title', f'Article {rec.get("article_id", "N/A")}')}</p>
                    <p>
                        <span class="category-tag">{rec.get('category', 'Unknown')}</span>
                    </p>
                    <div class="article-summary">
                        {rec.get('summary', 'No summary available')[:300]}...
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Interaction simulation section
    st.markdown('<p class="subtitle">Simulate User Interaction</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        interaction_article = st.selectbox(
            "Select Article", 
            options=recommendations['article_id'].tolist() if not recommendations.empty else []
        )
    
    with col2:
        interaction_type = st.selectbox(
            "Interaction Type",
            options=["click", "like", "bookmark"]  # Modified to match NewsFeedAgent accepted values
        )
    
    with col3:
        reading_time = st.slider(  # Changed from interaction_value to reading_time to match NewsFeedAgent
            "Reading Time (seconds)",
            min_value=5,
            max_value=600,
            value=60,
            step=5
        )
    
    if st.button("Submit Interaction"):
        if interaction_article is not None:
            with st.spinner("Updating user profile..."):
                # Simulate delay
                time.sleep(1)
                
                # Update user profile
                st.session_state.agent.update_with_feedback(  # Changed to match method in NewsFeedAgent
                    user_id=selected_user,
                    article_id=interaction_article,
                    interaction_type=interaction_type,
                    reading_time=reading_time
                )
                
                st.success(f"User profile updated! The new interaction with Article {interaction_article} has been recorded.")
                st.info("Generate recommendations again to see how they've changed based on this new interaction.")
        else:
            st.error("Please select an article for interaction.")

# Footer
st.markdown("---")
st.markdown("© 2025 PurpleMerit Personalized News Feed Agent")