import pandas as pd

# Define audio features columns in the desired order
AUDIO_FEATURES_COLUMNS = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo'
]

# Column Map
COLUMN_TO_FEATURE_MAP = {
    0: 'danceability',
    1: 'energy',
    2: 'speechiness',
    3: 'acousticness',
    4: 'instrumentalness',
    5: 'liveness',
    6: 'valence',
    7: 'loudness',
    8: 'tempo'
}

def normalize_features(df, columns):
    """
    Applies Z-score normalization to the specified columns.
    """
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            df[col] = (df[col] - mean_val) / std_val
        else:
            df[col] = 0.0 # Handle columns with zero variance
    return df

def load_and_preprocess_data(file_path='SpotifyCSV.csv'):
    """
    Loads the CSV, normalizes features, and removes duplicates.
    Returns the processed DataFrame and the features matrix.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file '{file_path}' not found.")
        print("Please download the 'Spotify Song Attribute Dataset' from Kaggle and save it as 'SpotifyCSV.csv'")
        return None, None
    
    # Z-score normalize [cite: 33]
    df = normalize_features(df, AUDIO_FEATURES_COLUMNS)
    
    # Drop duplicates
    df_processed = df.drop_duplicates(subset=['track_name', 'artist_names']).copy()
    
    # Reset index to ensure it's contiguous
    df_processed = df_processed.reset_index(drop=True)

    # Create audio features matrix without duplicates
    audio_features_matrix = df_processed[AUDIO_FEATURES_COLUMNS].values.tolist()
    
    print(f"Data loaded. {len(df_processed)} unique tracks found.")
    
    return df_processed, audio_features_matrix

if __name__ == "__main__":
    # For testing the preprocessing step
    df_tracks, features = load_and_preprocess_data()
    if df_tracks is not None:
        print("\n--- Column-to-Feature Mapping ---")
        for col, feature in COLUMN_TO_FEATURE_MAP.items():
            print(f"Column {col}: {feature}")
        
        print("\nProcessed DataFrame head:")
        print(df_tracks.head())
