from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import TfidfVectorizer
import numpy as np

class KNNRecommender:
    """
    A K-Nearest Neighbors recommender system based on audio features
    and optionally TF-IDF for artist/track names.
    """
    def __init__(self, k=10):
        self.k = k
        self.knn_model = NearestNeighbors(n_neighbors=self.k + 1, metric='cosine')
        self.df = None
        self.feature_matrix = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.combined_matrix = None

    def fit(self, df, audio_feature_matrix):
        """
        Fit the model with the processed data.
        """
        self.df = df
        self.feature_matrix = np.array(audio_feature_matrix)
        
        # As per the paper, KNN is fit on song audio features [cite: 13]
        self.knn_model.fit(self.feature_matrix)
        
        # --- Optional: TF-IDF for text features (as mentioned in paper [cite: 18-22]) ---
        # We can create a combined matrix of audio + text features
        
        # 1. TF-IDF for artist names
        self.tfidf_vectorizer = TfidfVectorizer()
        artist_tfidf = self.tfidf_vectorizer.fit_transform(self.df['artist_names'])
        
        # 2. Combine matrices
        # We'll simply concatenate the TF-IDF features with audio features
        # Note: This requires careful weighting, but we'll do a simple combine
        self.tfidf_matrix = artist_tfidf.toarray()
        
        # Ensure matrices have the same number of samples
        if self.feature_matrix.shape[0] == self.tfidf_matrix.shape[0]:
            self.combined_matrix = np.concatenate(
                [self.feature_matrix, self.tfidf_matrix], 
                axis=1
            )
            # Re-fit the model on the combined matrix
            self.knn_model.fit(self.combined_matrix)
            print("KNN model fit on COMBINED audio and text features.")
        else:
            # Fallback to just audio features
            self.combined_matrix = self.feature_matrix
            print("Warning: TF-IDF matrix shape mismatch. Fitting on AUDIO features only.")
        
    def _find_song_index(self, song_name, artist_name):
        """
        Find the index of a song in the dataframe.
        """
        if self.df is None:
            return None
            
        # Try to find an exact match
        result = self.df[(self.df['track_name'].str.lower() == song_name.lower()) & 
                         (self.df['artist_names'].str.lower().str.contains(artist_name.lower()))]
        
        if not result.empty:
            return result.index[0]
        
        # If no exact match, find the closest track name
        result = self.df[self.df['track_name'].str.lower() == song_name.lower()]
        if not result.empty:
            return result.index[0]
            
        return None

    def recommend(self, song_name, artist_name):
        """
        Recommend K similar songs based on a seed song.
        """
        song_index = self._find_song_index(song_name, artist_name)
        
        if song_index is None:
            return f"Error: Song '{song_name}' by '{artist_name}' not found."
            
        if self.combined_matrix is None:
            return "Error: Model not fit."

        # Get the feature vector for the seed song
        seed_vector = self.combined_matrix[song_index].reshape(1, -1)
        
        # Find neighbors
        distances, indices = self.knn_model.kneighbors(seed_vector)
        
        # Get the indices (skip the first one, it's the song itself)
        neighbor_indices = indices[0][1:]
        
        recommendations = self.df.iloc[neighbor_indices]
        
        return recommendations[['track_name', 'artist_names']]
