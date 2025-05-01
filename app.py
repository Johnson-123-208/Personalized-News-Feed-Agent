import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import json

# Add the project root to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your project modules
try:
    from agent import NewsFeedAgent
    from utils.data_loader import DataLoader
except ImportError:
    st.error("Failed to import required modules. Make sure your project structure is correct.")
    st.stop()

# Function to load Lottie animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Page configuration
st.set_page_config(
    page_title="Personalized News Feed Agent",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, animated interface
st.markdown("""
<style>
    /* Main layout */
    .main {
        background-color: #F8F9FD;
        padding: 2rem;
    }
    /* Text on light backgrounds (white or light gray cards) */
    .card, .user-profile-card, .article-card, .graph-container, .stat-card {
        color: #222 !important; /* dark gray for readability */
    }

    /* Text on dark backgrounds like sidebar */
    .css-1d391kg, .css-1oe6wy4 {  /* sidebar elements */
        background: linear-gradient(180deg, #5E35B1 0%, #4527A0 100%);
        color: white !important; /* make text readable */
    }

    /* Add default text-light and text-dark utility classes if needed in custom HTML */
    .text-light {
        color: white !important;
    }

    .text-dark {
        color: #222 !important;
    }

    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: # !important;
    }
    
    /* Header styling */
    .title {
        font-size: 42px !important;
        color: #D32F2F !important;
        font-weight: 700;
        background: linear-gradient(90deg, #5E35B1, #9575CD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
        animation: fadeIn 1.2s ease-out;
    }
    .card h2, .card h3 {
        color: #D32F2F !important;
    }

    
    .subtitle {
        font-size: 26px !important;
        color: #D32F2F !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-left: 4px solid #5E35B1;
        padding-left: 10px;
        animation: slideInLeft 0.8s ease-out;
    }
    
    /* Cards and containers */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.8s ease-out;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    .user-profile-card {
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(94, 53, 177, 0.1);
        margin-bottom: 2rem;
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Article styling */
    .article-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #5E35B1;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideInUp 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .article-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(94, 53, 177, 0.15);
    }
    
    .article-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        height: 4px;
        width: 0;
        background: linear-gradient(90deg, #5E35B1, #9575CD);
        transition: width 0.6s ease;
    }
    
    .article-card:hover::before {
        width: 100%;
    }
    
    .article-title {
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 0.8rem;
        color: #333;
    }
    
    .article-summary {
        font-size: 14px;
        color: #555;
        line-height: 1.6;
    }
    
    /* Category tags */
    .category-tag {
        background: linear-gradient(135deg, #E0D4FF, #D1C4F0);
        color: #5E35B1;
        padding: 0.3rem 0.7rem;
        border-radius: 50px;
        font-weight: 500;
        margin-right: 0.7rem;
        font-size: 12px;
        box-shadow: 0 2px 4px rgba(94, 53, 177, 0.1);
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    
    /* Score badge */
    .score-badge {
        background: linear-gradient(90deg, #5E35B1, #7E57C2);
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 50px;
        font-weight: 500;
        float: right;
        font-size: 12px;
        box-shadow: 0 2px 6px rgba(94, 53, 177, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #5E35B1, #7E57C2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        box-shadow: 0 4px 10px rgba(94, 53, 177, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(94, 53, 177, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 5px rgba(94, 53, 177, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1oe6wy4 {
        background: linear-gradient(180deg, #5E35B1 0%, #4527A0 100%);
    }
    
    .css-1v3fvcr {
        background-color: #F8F9FD;
    }
    
    /* Inputs and selectors */
    .stSelectbox, .stMultiSelect {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Progress and loaders */
    .stProgress > div > div {
        background-color: #7E57C2;
    }
    
    /* Toast messages */
    .element-container .stAlert {
        animation: slideInUp 0.5s ease-out;
    }
    
    /* Custom animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInUp {
        from { 
            opacity: 0;
            transform: translateY(20px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from { 
            opacity: 0;
            transform: translateX(-20px);
        }
        to { 
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(94, 53, 177, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(94, 53, 177, 0); }
        100% { box-shadow: 0 0 0 0 rgba(94, 53, 177, 0); }
    }
    
    /* Loading animation */
    .loading {
        text-align: center;
        padding: 2rem;
        animation: pulse 2s infinite;
    }
    
    /* Tab navigation */
    .tab-navigation {
        display: flex;
        margin-bottom: 2rem;
        overflow-x: auto;
        padding: 0.5rem 0;
        gap: 0.5rem;
    }
    
    .tab-button {
        background-color: white;
        color: #5E35B1;
        border: 1px solid #E0D4FF;
        border-radius: 50px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        white-space: nowrap;
    }
    
    .tab-button.active {
        background-color: #5E35B1;
        color: white;
        box-shadow: 0 4px 10px rgba(94, 53, 177, 0.2);
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .stat-value {
        font-size: 24px;
        font-weight: 600;
        color: #5E35B1;
    }
    
    .stat-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Graph containers */
    .graph-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        font-size: 14px;
        margin-top: 3rem;
        border-top: 1px solid #eee;
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
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "dashboard"
if 'loading_progress' not in st.session_state:
    st.session_state.loading_progress = 0
if 'recommendations_generated' not in st.session_state:
    st.session_state.recommendations_generated = False
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []

# Load animations
loading_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_t9gkkhz4.json")
news_animation = load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_qnpfavmd.json")
empty_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_0zny8xay.json")
success_animation = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_avngu8bm.json")

# Title with animation
st.markdown('<p class="title">Personalized News Feed Agent</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("PurpleMerit.png", width=200)
    st.markdown("## Control Panel")
    
    # Custom tab navigation
    tab_col1, tab_col2, tab_col3 = st.columns(3)
    with tab_col1:
        if st.button("Dashboard", key="tab_dashboard", 
                    help="View dashboard and analytics"):
            st.session_state.active_tab = "dashboard"
    with tab_col2:
        if st.button("Articles", key="tab_articles",
                    help="Browse recommended articles"):
            st.session_state.active_tab = "articles"
    with tab_col3:
        if st.button("Profile", key="tab_profile",
                    help="View user profile details"):
            st.session_state.active_tab = "profile"
    
    st.markdown("---")
    
    if not st.session_state.data_loaded:
        if st.button("✨ Load Data & Train Models", key="load_btn"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate loading progress with progress updates
            for i in range(101):
                # Update progress bar
                progress_bar.progress(i)
                
                # Update status message based on progress
                if i < 30:
                    status_text.text("Loading data files...")
                elif i < 60:
                    status_text.text("Preprocessing articles...")
                elif i < 90:
                    status_text.text("Training recommendation models...")
                else:
                    status_text.text("Finalizing setup...")
                
                # Slower at start and end for more natural feel
                sleep_time = 0.05
                if i < 10 or i > 90:
                    sleep_time = 0.1
                time.sleep(sleep_time)
            
            # Load data
            data_loader = DataLoader()
            st.session_state.articles, st.session_state.interactions = data_loader.load_data()
            
            # Initialize and train agent
            st.session_state.agent = NewsFeedAgent()
            st.session_state.agent.train(st.session_state.articles, st.session_state.interactions)
            
            # Get unique users
            st.session_state.users = sorted(st.session_state.interactions['user_id'].unique())
            st.session_state.data_loaded = True
            
            # Clear progress elements
            status_text.empty()
            
            # Show success animation
            if success_animation:
                st_lottie(success_animation, height=120, key="success_anim")
    
    if st.session_state.data_loaded:
        st.success("✅ Models trained and ready!")
        
        # User selection with search option
        st.markdown("### Select User")
        selected_user = st.selectbox(
            "User ID", 
            options=st.session_state.users,
            index=0,
            help="Select a user to generate personalized recommendations"
        )
        
        st.markdown("### Recommendation Settings")
        num_recommendations = st.slider(
            "Number of Recommendations", 
            min_value=3, 
            max_value=15, 
            value=5,
            help="Adjust the number of articles to recommend"
        )
        
        # Category filters with better styling
        if len(st.session_state.articles) > 0:
            st.markdown("### Content Filters")
            categories = ['All'] + sorted(st.session_state.articles['category'].unique().tolist())
            selected_categories = st.multiselect(
                "Filter by Category",
                options=categories[1:],
                default=[],
                help="Select specific categories to focus recommendations"
            )
        
        # Advanced settings with expander
        with st.expander("Advanced Settings", expanded=False):
            # Recommendation method with icons
            rec_method = st.radio(
                "Recommendation Method",
                options=["Hybrid", "Collaborative Filtering", "Content-Based"],
                index=0,
                help="Choose the algorithm approach for recommendations"
            )
            
            # Additional options
            exclude_seen = st.checkbox("Exclude Previously Seen Articles", value=True,
                                     help="Don't show articles the user has already interacted with")
            recency_boost = st.checkbox("Prioritize Recent Content", value=True,
                                      help="Give higher priority to newer articles")
            
        # Generate recommendations button with animation
        generate_col1, generate_col2 = st.columns([3, 1])
        with generate_col1:
            generate_btn = st.button("🚀 Generate Recommendations", key="gen_rec_btn",
                                   help="Generate personalized article recommendations for the selected user")
        
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application demonstrates a personalized news recommendation system using collaborative filtering and fine-tuned embeddings.
    
    [Documentation](https://example.com) | [GitHub](https://github.com)
    """)
    st.markdown("Created for PurpleMerit AI/ML Engineer Position")

# Main content area
if not st.session_state.data_loaded:
    # Welcome screen with animations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h2>Welcome to Personalized News Feed Agent</h2>
            <p>This intelligent system learns from user behavior to deliver the most relevant news articles tailored to individual interests and preferences.</p>
            <p>To get started, click the "Load Data & Train Models" button in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if news_animation:
            st_lottie(news_animation, height=200, key="welcome_anim")
    
    # Project description
    st.markdown("""
    <div class="card">
        <h2>Project Overview</h2>
        <p>This Personalized News Feed Agent combines collaborative filtering and fine-tuned embeddings to deliver tailored news recommendations to users based on their reading preferences and behaviors.</p>
        
        <div style="margin-top: 20px;">
            <h3>Key Features:</h3>
            <ul>
                <li><strong>Hybrid Recommendation Engine:</strong> Combines collaborative filtering with content-based approaches</li>
                <li><strong>User Profile Modeling:</strong> Builds detailed user profiles based on interaction history</li>
                <li><strong>Adaptive Learning:</strong> Updates recommendations based on user feedback</li>
                <li><strong>Category-Based Filtering:</strong> Allows filtering recommendations by topic categories</li>
            </ul>
        </div>
        
        <div style="margin-top: 20px;">
            <h3>Technologies Used:</h3>
            <ul>
                <li><strong>Collaborative Filtering:</strong> Identifies similar users and their content preferences</li>
                <li><strong>Doc2Vec Embeddings:</strong> Creates semantic representations of articles</li>
                <li><strong>Fine-tuned Models:</strong> Adapts pre-trained models to news domain</li>
                <li><strong>Streamlit:</strong> Powers this interactive web interface</li>
            </ul>
        </div>
    </div>""", unsafe_allow_html=True)
    
    # Sample visualization with modern Plotly charts
    st.markdown('<p class="subtitle">Sample Visualization</p>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    vis_tab1, vis_tab2 = st.tabs(["Category Distribution", "User Engagement"])
    
    with vis_tab1:
        # Sample data for visualization
        categories = ['Politics', 'Sports', 'Technology', 'Entertainment', 'Business']
        counts = [35, 25, 20, 15, 5]
        
        # Create modern Plotly bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=counts,
            marker_color=['#5E35B1', '#7E57C2', '#9575CD', '#B39DDB', '#D1C4E9'],
            text=counts,
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Category Distribution in Recommendations",
            xaxis_title="Category",
            yaxis_title="Number of Articles",
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with vis_tab2:
        # Sample data for user engagement over time
        dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
        engagement = [15, 22, 18, 25, 30, 35, 28, 32, 38, 42]
        
        # Create line chart
        fig = px.line(
            x=dates, y=engagement,
            labels={'x': 'Date', 'y': 'Number of Interactions'},
            title='User Engagement Trends',
        )
        
        fig.update_traces(line_color='#5E35B1', line_width=3)
        
        fig.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.data_loaded:
    # Handle generate button action
    if generate_btn:
        st.session_state.recommendations_generated = True
        
        # Show loading animation
        with st.spinner(""):
            if loading_animation:
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    st_lottie(loading_animation, height=200, key="loading_anim", speed=1.5)
            
            # Adapting method parameters to match NewsFeedAgent implementation
            method_param = {"Hybrid": None, "Collaborative Filtering": "cf_only", "Content-Based": "emb_only"}
            method = method_param.get(rec_method)
            
            # Generate recommendations
            recommendations = st.session_state.agent.get_recommendations(
                selected_user, 
                n=num_recommendations,
                exclude_seen=exclude_seen
            )
            
            # Filter by selected categories if any
            if selected_categories:
                recommendations = recommendations[recommendations['category'].isin(selected_categories)]
            
            # Store for session state
            st.session_state.current_recommendations = recommendations
            
            # Add to history for comparison
            timestamp = pd.Timestamp.now()
            history_entry = {
                'timestamp': timestamp,
                'user_id': selected_user,
                'num_recommendations': len(recommendations),
                'categories': selected_categories if selected_categories else ['All']
            }
            st.session_state.recommendation_history.append(history_entry)
            
            # Clear loading animation
            time.sleep(1)
            loading_placeholder.empty()
    
    # Get user profile for display
    if 'selected_user' in locals():
        user_profile = st.session_state.agent.get_user_profile(selected_user)
    else:
        # Default to first user if none selected
        selected_user = st.session_state.users[0]
        user_profile = st.session_state.agent.get_user_profile(selected_user)
    
    # Dashboard tab
    if st.session_state.active_tab == "dashboard":
        st.markdown('<p class="subtitle">Dashboard</p>', unsafe_allow_html=True)
        
        # User overview with statistics
        st.markdown("""
        <div class="user-profile-card">
        <h3>User Profile Overview</h3>
        """, unsafe_allow_html=True)
        
        # User stats in columns
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">User ID</div>
            </div>
            """.format(selected_user), unsafe_allow_html=True)
            
        with stats_cols[1]:
            total_interactions = user_profile.get('total_interactions', 0)
            st.markdown("""
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Interactions</div>
            </div>
            """.format(total_interactions), unsafe_allow_html=True)
            
        with stats_cols[2]:
            avg_time = user_profile.get('avg_reading_time', 0)
            st.markdown("""
            <div class="stat-card">
                <div class="stat-value">{:.1f}s</div>
                <div class="stat-label">Avg Reading Time</div>
            </div>
            """.format(avg_time), unsafe_allow_html=True)
            
        with stats_cols[3]:
            top_category = "Unknown"
            if 'category_preferences' in user_profile and user_profile['category_preferences']:
                top_category = max(user_profile['category_preferences'].items(), key=lambda x: x[1])[0]
            
            st.markdown("""
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Top Category</div>
            </div>
            """.format(top_category), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # User preference visualizations
        if 'category_preferences' in user_profile and user_profile['category_preferences']:
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            st.markdown("### User Interest Profile")
            
            # Convert category preferences to DataFrame for plotting
            categories = list(user_profile['category_preferences'].keys())
            weights = list(user_profile['category_preferences'].values())
            
            # Sort by weight
            sorted_data = sorted(zip(categories, weights), key=lambda x: x[1], reverse=True)
            categories = [item[0] for item in sorted_data]
            weights = [item[1] for item in sorted_data]
            
            # Create modern Plotly charts
            fig = px.bar(
                x=categories, 
                y=weights,
                labels={'x': 'Category', 'y': 'Preference Score'},
                color=weights,
                color_continuous_scale=px.colors.sequential.Purp,
                template="plotly_white"
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=40),
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Radar chart of interests 
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            st.markdown("### Interest Distribution")
            
            # Normalize for radar chart
            max_weight = max(weights) if weights else 1
            normalized_weights = [w/max_weight for w in weights]
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_weights,
                theta=categories,
                fill='toself',
                fillcolor='rgba(94, 53, 177, 0.2)',
                line=dict(color='#5E35B1', width=2),
                name='User Interests'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                template="plotly_white",
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendation history
        if st.session_state.recommendation_history:
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            st.markdown("### Recommendation History")
            
            history_df = pd.DataFrame(st.session_state.recommendation_history)
            history_df['time'] = history_df['timestamp'].dt.strftime('%H:%M:%S')
            
            fig = px.line(
                history_df, 
                x='time', 
                y='num_recommendations',
                markers=True,
                labels={'num_recommendations': 'Number of Recommendations', 'time': 'Time'},
            )
            
            fig.update_traces(line_color='#5E35B1', line_width=2)
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=40),
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Articles tab - show recommendations
    elif st.session_state.active_tab == "articles":
        st.markdown('<p class="subtitle">Recommended Articles</p>', unsafe_allow_html=True)
        
        if not st.session_state.recommendations_generated:
            # Show empty state with animation
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                <div class="card" style="text-align: center; padding: 2rem;">
                    <h3>No Recommendations Yet</h3>
                    <p>Click the "Generate Recommendations" button in the sidebar to see personalized article suggestions.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if empty_animation:
                    st_lottie(empty_animation, height=200, key="empty_anim")
        
        elif hasattr(st.session_state, 'current_recommendations'):
            recommendations = st.session_state.current_recommendations
            
            if len(recommendations) == 0:
                st.warning("No recommendations found matching the selected criteria. Try different filter settings.")
            else:
                # Filter controls
                sort_col1, sort_col2 = st.columns([1, 2])
                with sort_col1:
                    sort_by = st.selectbox("Sort by", ["Relevance", "Recent First", "Category"])
                
                with sort_col2:
                    view_mode = st.radio("View", ["Cards", "List"], horizontal=True)
                
                # Sort recommendations
                if sort_by == "Recent First":
                    recommendations = recommendations.sort_values(by='date', ascending=False) if 'date' in recommendations.columns else recommendations
                elif sort_by == "Category":
                    recommendations = recommendations.sort_values(by='category')
                
                # Display recommendations based on view mode
                if view_mode == "Cards":
                    # Card view - 2 columns
                    for i, (_, rec) in enumerate(recommendations.iterrows()):
                        st.markdown(
                            f"""
                            <div class="article-card">
                                <span class="score-badge">Score: {rec.get('score', 0):.2f}</span>
                                <p class="article-title">
                                    {i+1}. {rec.get('title', f'Article {rec.get("article_id", "N/A")}')}
                                </p>
                                <span class="category-tag">{rec.get('category', 'Unknown')}</span>
                                <div class="article-summary">
                                    {rec.get('summary', 'No summary available')[:200]}...
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    # List view - more compact
                    for i, (_, rec) in enumerate(recommendations.iterrows()):
                        with st.expander(f"{i+1}. {rec.get('title', f'Article {rec.get('article_id', 'N/A')}')} - {rec.get('category', 'Unknown')}"):
                            st.markdown(f"**Score:** {rec.get('score', 0):.4f}")
                            st.markdown(f"**Category:** {rec.get('category', 'Unknown')}")
                            st.markdown(f"**Summary:** {rec.get('summary', 'No summary available')}")
    
    # Profile tab - user profile and interaction
    elif st.session_state.active_tab == "profile":
        st.markdown('<p class="subtitle">User Profile</p>', unsafe_allow_html=True)
        
        # Detailed user profile
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="user-profile-card">
                <h3>User Details</h3>
                <p><strong>User ID:</strong> {}</p>
                <p><strong>Total interactions:</strong> {}</p>
                <p><strong>Average reading time:</strong> {:.2f} seconds</p>
            </div>
            """.format(
                selected_user,
                user_profile.get('total_interactions', 0),
                user_profile.get('avg_reading_time', 0)
            ), unsafe_allow_html=True)
            
            # Category preferences
            if 'category_preferences' in user_profile and user_profile['category_preferences']:
                st.markdown("""
                <div class="card">
                    <h3>Category Preferences</h3>
                """, unsafe_allow_html=True)
                
                # Convert to percentage for better understanding
                total = sum(user_profile['category_preferences'].values())
                for category, weight in sorted(user_profile['category_preferences'].items(), key=lambda x: x[1], reverse=True):
                    percentage = (weight / total) * 100 if total > 0 else 0
                    
                    # Progress bar for each category
                    st.markdown(f"**{category}**")
                    st.progress(min(percentage/100, 1.0))
                    st.markdown(f"<div style='text-align: right; color: #666; font-size: 14px;'>{percentage:.1f}%</div>", 
                                unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # User avatar and stats
            st.markdown("""
            <div class="card" style="text-align: center;">
                <img src="https://api.dicebear.com/7.x/initials/svg?seed={}" style="width: 120px; border-radius: 50%; margin-bottom: 15px;">
                <h3>User {}</h3>
                <p style="color: #666;">Member since 2025</p>
            </div>
            """.format(selected_user, selected_user), unsafe_allow_html=True)
            
            # Reading activity chart
            st.markdown("""
            <div class="card">
                <h3>Reading Activity</h3>
            """, unsafe_allow_html=True)
            
            # Sample data for reading activity
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            activity = [4, 6, 3, 7, 8, 5, 2]
            
            fig = px.bar(
                x=days, 
                y=activity,
                labels={'x': 'Day', 'y': 'Articles Read'},
                color=activity,
                color_continuous_scale=px.colors.sequential.Purp,
            )
            
            fig.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_white",
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Interaction simulation section
        st.markdown('<p class="subtitle">Simulate User Interaction</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Log New Interaction</h3>
            <p>Record a new user interaction to see how it affects recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Show recommended articles for interaction if available
            if hasattr(st.session_state, 'current_recommendations') and not st.session_state.current_recommendations.empty:
                article_options = st.session_state.current_recommendations['article_id'].tolist()
                article_titles = st.session_state.current_recommendations['title'].tolist() if 'title' in st.session_state.current_recommendations.columns else [f"Article {id}" for id in article_options]
                
                # Create a dictionary mapping article IDs to their titles for display
                article_display = {str(aid): title for aid, title in zip(article_options, article_titles)}
                
                interaction_article = st.selectbox(
                    "Select Article",
                    options=[str(id) for id in article_options],
                    format_func=lambda x: article_display.get(x, x)
                )
            else:
                # Fallback if no recommendations are available
                interaction_article = st.text_input("Article ID")
        
        with col2:
            interaction_type = st.selectbox(
                "Interaction Type",
                options=["click", "like", "bookmark", "share", "comment"],
                help="Type of interaction with the article"
            )
        
        with col3:
            reading_time = st.slider(
                "Reading Time (seconds)",
                min_value=5,
                max_value=600,
                value=60,
                step=5,
                help="Time spent reading the article"
            )
        
        # Additional feedback options
        feedback_col1, feedback_col2 = st.columns(2)
        
        with feedback_col1:
            relevance_rating = st.slider(
                "Relevance Rating",
                min_value=1,
                max_value=5,
                value=3,
                help="How relevant was this article to the user"
            )
        
        with feedback_col2:
            sentiment = st.radio(
                "Sentiment",
                options=["Positive", "Neutral", "Negative"],
                index=0,
                horizontal=True,
                help="User's reaction to the article"
            )
        
        # Submit button with improved styling
        submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
        with submit_col2:
            if st.button("📝 Submit Interaction", key="submit_interaction"):
                if interaction_article:
                    # Display visually appealing progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create animated progress
                    for i in range(101):
                        progress_bar.progress(i)
                        
                        if i < 50:
                            status_text.text("Processing interaction...")
                        else:
                            status_text.text("Updating user profile...")
                        
                        time.sleep(0.01)
                    
                    # Convert string ID back to original type if needed
                    try:
                        interaction_article_id = int(interaction_article)
                    except ValueError:
                        interaction_article_id = interaction_article
                    
                    # Update user profile
                    st.session_state.agent.update_with_feedback(
                        user_id=selected_user,
                        article_id=interaction_article_id,
                        interaction_type=interaction_type,
                        reading_time=reading_time
                    )
                    
                    # Clear status elements
                    status_text.empty()
                    
                    # Success message with animation
                    st.success("User profile updated! The interaction has been recorded.")
                    if success_animation:
                        st_lottie(success_animation, height=120, key="success_interaction")
                    
                    st.info("Generate recommendations again to see how they've changed based on this new interaction.")
                else:
                    st.error("Please select an article for interaction.")

# Footer with improved styling
st.markdown("""
<div class="footer">
    <p>© 2025 PurpleMerit Personalized News Feed Agent</p>
    <p style="font-size: 12px; color: #999;">Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)