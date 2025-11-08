# This file creates the SVD/ALS recommender.

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

class SVDRecommender:
    """
    A recommender system based on SVD (Matrix Factorization)
    using the 'Surprise' library as mentioned in the paper .
    
    This model predicts user ratings for songs.
    """
    def __init__(self):
        self.model = SVD(n_factors=50, n_epochs=20, random_state=42)
        self.trainset = None
        self.all_songs_df = None

    def fit(self, df_tracks):
        """
        Fit the SVD model.
        
        Since the dataset doesn't have users, we will *simulate*
        user-item interactions (e.g., 'listens' or 'ratings')
        for demonstration purposes.
        """
        self.all_songs_df = df_tracks
        
        # --- Simulate User Data ---
        # Create 100 mock users
        num_users = 100
        num_ratings_per_user = 50
        
        ratings_list = []
        
        # Give each user some random ratings
        for user_id in range(num_users):
            # Pick 50 random songs for this user
            user_songs = df_tracks.sample(n=num_ratings_per_user)
            for idx, row in user_songs.iterrows():
                # Give a random rating (1-5)
                rating = np.random.randint(1, 6)
                ratings_list.append({
                    'userID': f'user_{user_id}',
                    'songID': row['track_id'], # Assuming 'track_id' exists
                    'rating': rating
                })
        
        if not ratings_list:
            print("Error: Could not simulate ratings.")
            return

        ratings_df = pd.DataFrame(ratings_list)
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['userID', 'songID', 'rating']], reader)
        
        # Build full training set
        self.trainset = data.build_full_trainset()
        
        print("\nFitting SVD model on simulated user data...")
        self.model.fit(self.trainset)
        print("SVD model fit.")

    def get_top_n_recommendations(self, user_id, n=10):
        """
        Get Top-N recommendations for a specific user.
        """
        if self.trainset is None:
            return "Error: Model not fit."

        # Get a list of all song IDs
        all_song_iids = self.trainset.all_items()
        
        # Get song IDs the user has *already* rated
        try:
            user_inner_id = self.trainset.to_inner_uid(user_id)
            rated_song_iids = {item_iid for (item_iid, _) in self.trainset.ur[user_inner_id]}
        except ValueError:
            return f"Error: User '{user_id}' not found in training data."
        
        # Predict ratings for songs the user has *not* rated
        predictions = []
        for song_iid in all_song_iids:
            if song_iid not in rated_song_iids:
                # Convert back to raw songID
                raw_song_id = self.trainset.to_raw_iid(song_iid)
                pred = self.model.predict(user_id, raw_song_id)
                predictions.append((raw_song_id, pred.est))
        
        # Sort predictions by estimated rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_n_raw_ids = [song_id for (song_id, rating) in predictions[:n]]
        
        # Map raw IDs back to song names
        top_n_songs = self.all_songs_df[self.all_songs_df['track_id'].isin(top_n_raw_ids)]
        
        return top_n_songs[['track_name', 'artist_names']]
