import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Load data and prepare the recommender model
df = pd.read_csv("anime.csv")
df["genre"] = df["genre"].fillna("")  # Replace NaN with empty string
df["genre"] = df["genre"].astype(str)  # Ensure all values are strings

# TF-IDF Vectorization on genres
vectorizer = TfidfVectorizer(stop_words="english")
genre_matrix = vectorizer.fit_transform(df["genre"])

# Compute Cosine Similarity Matrix for genres
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/recommend', methods=['GET'])
def recommend_anime_api():
    title = request.args.get('title', '')
    count = int(request.args.get('count', 6))
    
    if not title:
        return jsonify({"error": "No anime title provided"}), 400
    
    # Case insensitive title search - more likely to find matches
    matching_rows = df[df["name"].str.lower() == title.lower()]
    if matching_rows.empty:
        # Try partial match if exact match fails
        matching_rows = df[df["name"].str.lower().str.contains(title.lower())]
        if matching_rows.empty:
            return jsonify({"error": f"Anime '{title}' not found in the dataset."}), 404
    
    # Get index of the given anime (use first match)
    idx = matching_rows.index[0]
    
    # Get the type of the anime (TV, Movie, OVA, etc.)
    anime_type = df.loc[idx, "type"]
    
    # Get similarity scores for all animes
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Filter only animes with the same type
    filtered_animes = [
        (i, score) for i, score in sim_scores if df.loc[i, "type"] == anime_type
    ]
    
    # Sort by similarity score (descending)
    filtered_animes = sorted(filtered_animes, key=lambda x: x[1], reverse=True)
    
    # Get top similar animes (excluding the input anime itself)
    top_animes = filtered_animes[1:count+1]  # Skip first item (itself)
    
    # If we don't have enough recommendations, include other types as well
    if len(top_animes) < count:
        remaining_needed = count - len(top_animes)
        # Get remaining from other types
        other_types = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        other_types = [(i, score) for i, score in other_types if df.loc[i, "type"] != anime_type]
        top_animes.extend(other_types[:remaining_needed])
    
    # Retrieve anime indices
    anime_indices = [i[0] for i in top_animes]
    
    # Get recommended animes data
    recommended_animes = df.iloc[anime_indices].sort_values(by="rating", ascending=False)
    
    # Convert to list of dicts for JSON response
    result = recommended_animes[["name", "genre", "type", "rating"]].to_dict(orient="records")
    
    return jsonify(result)

@app.route('/api/search', methods=['GET'])
def search_anime():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    
    # Search for anime with names containing the query (case insensitive)
    matches = df[df["name"].str.lower().str.contains(query.lower())]
    # Return top 10 matches
    result = matches.head(10)[["name", "genre", "type", "rating"]].to_dict(orient="records")
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
