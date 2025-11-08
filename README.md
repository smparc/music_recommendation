# Music Recommendation Engine

This project implements a music recommendation engine based on the paper "Music Recommendation Engine: A Hybrid Approach." It uses the **Spotify Song Attribute Dataset** from Kaggle and builds two types of models:

1.  [cite_start]**Content-Based (KNN):** Recommends songs based on similar audio features (danceability, energy, etc.) and artist information .
2.  [cite_start]**Collaborative Filtering (SVD):** Recommends songs based on simulated user-item ratings, using the "Surprise" library .

## 🚀 How to Use

### 1. Get the Data
This project requires the **Spotify Song Attribute Dataset** from Kaggle.
1.  Download the dataset (it's often a zip file).
2.  Find the file containing song attributes (e.g., `SpotifyCSV.csv` or `tracks.csv`).
3.  **Important:** Rename this file to `SpotifyCSV.csv` and place it in the root of this project folder.
4.  **Requirement:** The CSV *must* contain a `track_id` column for the SVD model to work.

### 2. Setup
Create a Python environment and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Project
Run the script:
```bash
python main.py
```

This will:
1. Load and preprocess SpotifyCSV.csv, normalizing features and removing duplicates
2. Train the KNN model and generate recommendations for a few example songs
3. Train the SVD model (on simulated used data) and generate recommendations for a sample user
