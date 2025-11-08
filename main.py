import data_preprocessing
from knn_model import KNNRecommender
from svd_model import SVDRecommender

def run_knn_demo(df, features):
    """
    Demonstrates the KNN model.
    """
    print("\n--- 🎵 KNN Recommender Demo ---")
    knn = KNNRecommender(k=10)
    knn.fit(df, features)
    
    # --- Get Recommendations ---
    # Seed song from your paper
    song_name = "Smells Like Teen Spirit"
    artist_name = "Nirvana"
    
    print(f"\nRecommendations for '{song_name}' by '{artist_name}':")
    recommendations = knn.recommend(song_name, artist_name)
    print(recommendations)
    
    # Another example
    song_name = "Hey Jude"
    artist_name = "The Beatles"
    
    print(f"\nRecommendations for '{song_name}' by '{artist_name}':")
    recommendations = knn.recommend(song_name, artist_name)
    print(recommendations)

def run_svd_demo(df):
    """
    Demonstrates the SVD model.
    """
    if 'track_id' not in df.columns:
        print("\nError: 'track_id' column not found, which is needed for SVD.")
        print("Please ensure your 'SpotifyCSV.csv' has a 'track_id' column.")
        return
        
    print("\n\n--- 🤖 SVD Recommender Demo ---")
    svd = SVDRecommender()
    svd.fit(df)
    
    # --- Get Recommendations ---
    # This uses one of our simulated users
    user = 'user_5'
    
    print(f"\nTop 10 recommendations for {user} (simulated):")
    recommendations = svd.get_top_n_recommendations(user, n=10)
    print(recommendations)

def main():
    # Load and process data first
    df, features = data_preprocessing.load_and_preprocess_data()
    
    if df is None:
        return

    # Run KNN Demo
    run_knn_demo(df, features)
    
    # Run SVD Demo
    run_svd_demo(df)

if __name__ == "__main__":
    main()
