import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import logging

# Setup basic logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class EmbeddingModel:
    """
    News article embedding model with fine-tuning capabilities
    """
    
    def __init__(self, embedding_size=100, window=5, min_count=2):
        """
        Initialize the embedding model
        
        Args:
            embedding_size: Size of the embedding vectors
            window: Window size for Word2Vec context
            min_count: Minimum frequency for words to be included
        """
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.article_embeddings = {}
        self.user_embeddings = {}
        
        # Download NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_text(self, text):
        """Preprocess text for embedding"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
        
        return tokens
    
    def fit(self, articles_df, interactions_df=None):
        """
        Train the embedding model on article content
        
        Args:
            articles_df: DataFrame with article data
            interactions_df: Optional DataFrame with user interactions for user embedding creation
        """
        # Preprocess and tokenize articles
        print("Preprocessing articles...")
        processed_articles = []
        
        for idx, article in articles_df.iterrows():
            tokens = self.preprocess_text(article['content'])
            if tokens:  # Only include non-empty documents
                doc = TaggedDocument(words=tokens, tags=[f"DOC_{article['article_id']}"])
                processed_articles.append(doc)
        
        # Train Doc2Vec model
        print("Training Doc2Vec model...")
        self.model = Doc2Vec(
            vector_size=self.embedding_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=20
        )
        
        # Build vocabulary
        self.model.build_vocab(processed_articles)
        
        # Train the model
        self.model.train(
            processed_articles,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs
        )
        
        # Generate article embeddings
        print("Generating article embeddings...")
        for article_id in articles_df['article_id'].values:
            article_tag = f"DOC_{article_id}"
            if article_tag in self.model.dv:
                self.article_embeddings[article_id] = self.model.dv[article_tag]
        
        # Generate user embeddings if interaction data is provided
        if interactions_df is not None:
            print("Generating user embeddings...")
            self._generate_user_embeddings(interactions_df)
    
    def _generate_user_embeddings(self, interactions_df):
        """
        Generate user embeddings based on their interaction history
        
        Args:
            interactions_df: DataFrame with user interaction data
        """
        # Group interactions by user
        user_interactions = interactions_df.groupby('user_id')
        
        # Generate user embeddings as weighted average of article embeddings
        for user_id, interactions in user_interactions:
            # Get articles this user has interacted with
            article_weights = {}
            
            # Calculate weights based on interaction type and reading time
            for _, interaction in interactions.iterrows():
                article_id = interaction['article_id']
                
                # Skip if we don't have an embedding for this article
                if article_id not in self.article_embeddings:
                    continue
                
                # Calculate interaction weight
                weight = interaction['reading_time']
                if interaction['interaction_type'] == 'like':
                    weight *= 2
                elif interaction['interaction_type'] == 'bookmark':
                    weight *= 3
                
                # Update weight (take maximum if multiple interactions)
                if article_id in article_weights:
                    article_weights[article_id] = max(article_weights[article_id], weight)
                else:
                    article_weights[article_id] = weight
            
            # Skip users with no valid interactions
            if not article_weights:
                continue
            
            # Calculate weighted average embedding
            total_weight = sum(article_weights.values())
            user_embedding = np.zeros(self.embedding_size)
            
            for article_id, weight in article_weights.items():
                user_embedding += (weight / total_weight) * self.article_embeddings[article_id]
            
            # Store user embedding
            self.user_embeddings[user_id] = user_embedding
    
    def fine_tune(self, interactions_df, epochs=5):
        """
        Fine-tune the embedding model based on user interactions
        
        Args:
            interactions_df: DataFrame with user interaction data
            epochs: Number of fine-tuning epochs
        """
        if self.model is None:
            raise ValueError("Model needs to be trained first before fine-tuning")
        
        # Group positive interactions (likes, bookmarks) to create pairs for fine-tuning
        positive_interactions = interactions_df[
            (interactions_df['interaction_type'] == 'like') | 
            (interactions_df['interaction_type'] == 'bookmark')
        ]
        
        # Group by user to get article pairs
        user_article_groups = positive_interactions.groupby('user_id')['article_id'].apply(list)
        
        # Create positive article pairs from same user interactions
        article_pairs = []
        for user_articles in user_article_groups:
            if len(user_articles) < 2:
                continue
                
            # Create pairs of articles that the same user liked
            for i in range(len(user_articles)):
                for j in range(i+1, len(user_articles)):
                    article_pairs.append((user_articles[i], user_articles[j]))
        
        # Fine-tune the model with these pairs
        if article_pairs:
            print(f"Fine-tuning with {len(article_pairs)} article pairs...")
            
            # Adjust the learning rate for fine-tuning
            original_alpha = self.model.alpha
            self.model.alpha = original_alpha / 10
            
            # Fine-tune for specified number of epochs
            for epoch in range(epochs):
                np.random.shuffle(article_pairs)
                
                for article1_id, article2_id in article_pairs:
                    # Get the document tags
                    tag1 = f"DOC_{article1_id}"
                    tag2 = f"DOC_{article2_id}"
                    
                    # Skip if either article is not in the vocabulary
                    if tag1 not in self.model.dv or tag2 not in self.model.dv:
                        continue
                    
                    # Get current embeddings
                    vec1 = self.model.dv[tag1]
                    vec2 = self.model.dv[tag2]
                    
                    # Move embeddings slightly closer to each other
                    target = (vec1 + vec2) / 2
                    
                    # Update embeddings
                    self.model.dv.vectors[self.model.dv.key_to_index[tag1]] += 0.01 * (target - vec1)
                    self.model.dv.vectors[self.model.dv.key_to_index[tag2]] += 0.01 * (target - vec2)
                    
                    # Normalize vectors
                    self.model.dv.vectors[self.model.dv.key_to_index[tag1]] /= np.linalg.norm(
                        self.model.dv.vectors[self.model.dv.key_to_index[tag1]]
                    )
                    self.model.dv.vectors[self.model.dv.key_to_index[tag2]] /= np.linalg.norm(
                        self.model.dv.vectors[self.model.dv.key_to_index[tag2]]
                    )
                
                print(f"Completed epoch {epoch+1}/{epochs}")
            
            # Reset learning rate
            self.model.alpha = original_alpha
            
            # Update article embeddings after fine-tuning
            for article_id in self.article_embeddings:
                article_tag = f"DOC_{article_id}"
                if article_tag in self.model.dv:
                    self.article_embeddings[article_id] = self.model.dv[article_tag]
            
            # Update user embeddings
            self._generate_user_embeddings(interactions_df)
    
    def recommend(self, user_id, articles_df, n=10, exclude_seen=True, interactions_df=None):
        """
        Recommend articles for a user based on embedding similarity
        
        Args:
            user_id: ID of the user to recommend articles to
            articles_df: DataFrame with article data
            n: Number of recommendations to generate
            exclude_seen: Whether to exclude articles the user has already interacted with
            interactions_df: Optional DataFrame with user interactions to exclude seen articles
            
        Returns:
            List of recommended article IDs
        """
        # Check if we have an embedding for this user
        if user_id not in self.user_embeddings:
            if interactions_df is not None:
                # Try to generate an embedding based on recent interactions
                user_interactions = interactions_df[interactions_df['user_id'] == user_id]
                if not user_interactions.empty:
                    self._generate_user_embeddings(user_interactions)
            
            # If still no embedding, return empty recommendations
            if user_id not in self.user_embeddings:
                return []
        
        # Get user embedding
        user_embedding = self.user_embeddings[user_id]
        
        # Get articles the user has already seen
        seen_articles = set()
        if exclude_seen and interactions_df is not None:
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            seen_articles = set(user_interactions['article_id'].values)
        
        # Calculate similarity between user embedding and article embeddings
        similarities = []
        
        for article_id in articles_df['article_id'].values:
            # Skip if article doesn't have an embedding or user has seen it
            if article_id not in self.article_embeddings or (exclude_seen and article_id in seen_articles):
                continue
                
            # Calculate cosine similarity
            article_embedding = self.article_embeddings[article_id]
            similarity = cosine_similarity([user_embedding], [article_embedding])[0][0]
            
            similarities.append((article_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations
        recommendations = [article_id for article_id, _ in similarities[:n]]
        
        return recommendations
    
    def get_article_embedding(self, article_id):
        """Get the embedding for a specific article"""
        return self.article_embeddings.get(article_id, None)
    
    def get_user_embedding(self, user_id):
        """Get the embedding for a specific user"""
        return self.user_embeddings.get(user_id, None)
    
    def update_user_embedding(self, user_id, interactions_df):
        """Update a user's embedding with new interaction data"""
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        if not user_interactions.empty:
            self._generate_user_embeddings(user_interactions)