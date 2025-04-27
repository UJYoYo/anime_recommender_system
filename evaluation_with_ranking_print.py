import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import logging
import os
import datetime

# Set up logging to file
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"recommender_evaluation_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to console
    ]
)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in')

# Load the datasets
try:
    anime_df = pd.read_csv('anime.csv')
    ratings_df = pd.read_csv('rating.csv')
    logging.info(f"Successfully loaded datasets. Anime dataset shape: {anime_df.shape}, Ratings dataset shape: {ratings_df.shape}")
except Exception as e:
    logging.error(f"Error loading datasets: {e}")
    # Create a small sample dataset if files can't be loaded
    anime_df = pd.read_csv('anime_cleaned.csv')
    # Create an empty ratings dataframe with the expected structure
    ratings_df = pd.DataFrame(columns=['user_id', 'anime_id', 'rating'])
    logging.info(f"Using sample anime dataset with shape: {anime_df.shape}")

# Clean the data - same preprocessing as in your notebook
anime_df['genre'] = anime_df['genre'].fillna('')
anime_df['genre'] = anime_df['genre'].astype(str)

# Create TF-IDF matrix for genre similarity - same as your implementation
vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = vectorizer.fit_transform(anime_df['genre'])

# Compute Cosine Similarity Matrix for genres
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
logging.info("Cosine similarity matrix computed successfully")

# Your exact recommendation function from the provided code
def recommend_anime(title, top_n=5):
    """
    Your exact recommendation function from the provided code.
    Modified to handle boolean array issues with proper error handling.
    """
    # Check if anime exists
    if title not in anime_df["name"].values:
        return f"Anime '{title}' not found in the dataset."

    try:
        # Get index of the given anime
        idx = anime_df[anime_df["name"] == title].index[0]

        # Get the type of the anime (TV, Movie, OVA, etc.)
        anime_type = anime_df.loc[idx, "type"]

        # Get similarity scores for all animes
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Filter only animes with the same type - with improved error handling
        filtered_animes = []
        for i, score in sim_scores:
            if i < len(anime_df):
                try:
                    db_type = anime_df.loc[i, "type"]
                    # Handle case where comparison might return an array
                    if isinstance(db_type, str) and db_type == anime_type:
                        filtered_animes.append((i, score))
                except Exception as e:
                    continue

        # Sort by similarity score (descending)
        filtered_animes = sorted(filtered_animes, key=lambda x: x[1], reverse=True)

        # Get top similar animes (excluding the input anime itself)
        top_animes = filtered_animes[1:top_n + 1]  # Skip first item (itself)
        
        if not top_animes:
            return pd.DataFrame(columns=["name", "genre", "type", "rating"])

        # Retrieve anime indices
        anime_indices = [i[0] for i in top_animes]

        # Get recommended animes and sort by rating (descending)
        recommended_animes = anime_df.iloc[anime_indices].sort_values(by="rating", ascending=False)

        return recommended_animes[["name", "genre", "type", "rating"]]
    except Exception as e:
        logging.error(f"Error in recommendation for '{title}': {e}")
        return f"Error recommending anime similar to '{title}'"

# Helper functions for evaluation
def safe_mean(values):
    """Calculate mean after filtering out None values and NaN values"""
    filtered_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if filtered_values:
        return np.mean(filtered_values)
    else:
        return None

def safe_corrcoef(x, y):
    """Calculate correlation coefficient with proper error handling"""
    # Check if inputs are empty
    if not x or not y:
        return None
        
    # Convert to numpy arrays
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    
    # Remove any NaN or infinite values
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr) | np.isinf(x_arr) | np.isinf(y_arr))
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    
    # Check if we have enough data and variation
    if len(x_clean) < 2 or np.std(x_clean) == 0 or np.std(y_clean) == 0:
        return None
    
    return np.corrcoef(x_clean, y_clean)[0, 1]

# Evaluation functions
def evaluate_similarity_quality(sample_size=1000):
    """
    Evaluate if the TF-IDF cosine similarity aligns with direct genre overlap
    """
    logging.info(f"Starting similarity quality evaluation with sample size: {sample_size}")
    n_anime = len(anime_df)
    
    # Sample random pairs for efficiency
    pairs = []
    for _ in range(min(sample_size, n_anime * (n_anime - 1) // 2)):
        i = np.random.randint(0, n_anime)
        j = np.random.randint(0, n_anime)
        if i != j and (i, j) not in pairs and (j, i) not in pairs:
            pairs.append((i, j))
    
    # Calculate genre overlap and similarity for each pair
    jaccard_similarities = []
    cosine_similarities = []
    
    for count, (i, j) in enumerate(pairs):
        # Log progress periodically
        if count % 100 == 0:
            logging.debug(f"Processing similarity pair {count}/{len(pairs)}")
        
        # Calculate Jaccard similarity (direct genre overlap)
        genres_i = set(anime_df.iloc[i]['genre'].split(', ')) if anime_df.iloc[i]['genre'] else set()
        genres_j = set(anime_df.iloc[j]['genre'].split(', ')) if anime_df.iloc[j]['genre'] else set()
        
        # Remove empty strings from the sets
        genres_i = {g for g in genres_i if g}
        genres_j = {g for g in genres_j if g}
        
        if not genres_i or not genres_j:
            continue
            
        intersection = len(genres_i.intersection(genres_j))
        union = len(genres_i.union(genres_j))
        jaccard = intersection / union if union > 0 else 0
        
        # Get cosine similarity from matrix
        cosine = cosine_sim[i, j]
        
        jaccard_similarities.append(jaccard)
        cosine_similarities.append(cosine)
    
    # Calculate correlation between Jaccard and Cosine similarity
    correlation = safe_corrcoef(jaccard_similarities, cosine_similarities) if jaccard_similarities else None
    
    results = {
        'Correlation': correlation,
        'Jaccard Mean': safe_mean(jaccard_similarities),
        'Cosine Mean': safe_mean(cosine_similarities),
        'Sample Size': len(jaccard_similarities)
    }
    
    logging.info(f"Similarity quality evaluation completed: {results}")
    return results

def evaluate_ranking_quality(num_samples=50):
    """
    Evaluate ranking metrics for the recommender system with detailed error logging
    """
    logging.info(f"Starting ranking quality evaluation with {num_samples} samples")
    
    # Sample anime to evaluate
    sample_indices = np.random.choice(range(len(anime_df)), min(num_samples, len(anime_df)), replace=False)
    
    # Initialize metrics
    precision_at_k = []  # Precision@K
    ndcg_scores = []     # Normalized Discounted Cumulative Gain
    
    processed_count = 0
    skipped_anime = []  # Track skipped anime and reasons
    
    for idx in sample_indices:
        try:
            title = anime_df.iloc[idx]['name']
            anime_id = anime_df.iloc[idx]['anime_id']
            
            # Get the genres safely
            input_genres_str = anime_df.iloc[idx]['genre']
            if pd.isna(input_genres_str) or not isinstance(input_genres_str, str) or input_genres_str == '':
                skipped_anime.append({
                    'anime_id': anime_id,
                    'title': title,
                    'reason': 'Invalid or empty genre information'
                })
                continue
                
            input_genres = set(input_genres_str.split(', '))
            input_genres = {g for g in input_genres if g}
            
            if not input_genres:
                skipped_anime.append({
                    'anime_id': anime_id,
                    'title': title,
                    'reason': 'No valid genres after processing'
                })
                continue
            
            # Get recommendations
            recommendations = recommend_anime(title, top_n=10)
            
            if isinstance(recommendations, str):  # Error message
                skipped_anime.append({
                    'anime_id': anime_id,
                    'title': title,
                    'reason': f'Recommendation error: {recommendations}'
                })
                continue
                
            if len(recommendations) == 0:
                skipped_anime.append({
                    'anime_id': anime_id,
                    'title': title,
                    'reason': 'No recommendations found'
                })
                continue
            
            # Calculate Precision@K
            # (What fraction of recommended items are relevant?)
            relevant_count = 0
            for _, row in recommendations.iterrows():
                try:
                    rec_genres_str = row['genre']
                    if pd.isna(rec_genres_str) or not isinstance(rec_genres_str, str) or rec_genres_str == '':
                        continue
                        
                    rec_genres = set(rec_genres_str.split(', '))
                    rec_genres = {g for g in rec_genres if g}
                    
                    # An item is considered relevant if it shares at least one genre
                    if input_genres.intersection(rec_genres):
                        relevant_count += 1
                except Exception as e:
                    logging.error(f"Error processing genre in Precision calculation: {e}")
                    continue
            
            precision = relevant_count / len(recommendations)
            precision_at_k.append(precision)
            
            # Calculate NDCG (Normalized Discounted Cumulative Gain)
            try:
                # Calculate relevance scores
                relevance_scores = []
                for _, row in recommendations.iterrows():
                    try:
                        rec_genres_str = row['genre']
                        if pd.isna(rec_genres_str) or not isinstance(rec_genres_str, str) or rec_genres_str == '':
                            relevance_scores.append(0)
                            continue
                            
                        rec_genres = set(rec_genres_str.split(', '))
                        rec_genres = {g for g in rec_genres if g}
                        
                        # Calculate relevance as Jaccard similarity between genre sets
                        if rec_genres:
                            intersection = len(input_genres.intersection(rec_genres))
                            union = len(input_genres.union(rec_genres))
                            relevance = intersection / union if union > 0 else 0
                        else:
                            relevance = 0
                        
                        relevance_scores.append(relevance)
                    except Exception as e:
                        logging.error(f"Error calculating relevance score: {e}")
                        relevance_scores.append(0)
                
                # Calculate DCG (Discounted Cumulative Gain)
                if not relevance_scores:
                    continue
                    
                dcg = relevance_scores[0]
                for i in range(1, len(relevance_scores)):
                    # Use log base 2 for position discount
                    position_discount = np.log2(i + 2)  # +2 because i starts at 1 and log2(2) = 1
                    if position_discount > 0:
                        dcg += relevance_scores[i] / position_discount
                
                # Calculate ideal DCG (sorted relevance scores)
                ideal_relevance_scores = sorted(relevance_scores, reverse=True)
                idcg = ideal_relevance_scores[0]
                for i in range(1, len(ideal_relevance_scores)):
                    position_discount = np.log2(i + 2)
                    if position_discount > 0:
                        idcg += ideal_relevance_scores[i] / position_discount
                
                # Calculate NDCG
                if idcg > 0:
                    ndcg = dcg / idcg
                    ndcg_scores.append(ndcg)
            except Exception as e:
                logging.error(f"Error calculating NDCG: {e}")
                continue
            
            processed_count += 1
            logging.debug(f"Successfully processed anime {processed_count}/{num_samples}: {title}")
            
        except Exception as e:
            skipped_anime.append({
                'anime_id': anime_df.iloc[idx]['anime_id'] if idx < len(anime_df) else 'unknown',
                'title': anime_df.iloc[idx]['name'] if idx < len(anime_df) else f'index {idx}',
                'reason': f'Error: {str(e)}'
            })
            continue
    
    # Log the skipped anime details
    if skipped_anime:
        logging.info(f"Details of {len(skipped_anime)} skipped anime in ranking evaluation:")
        for i, anime in enumerate(skipped_anime, 1):
            logging.info(f"  {i}. ID: {anime['anime_id']}, Title: '{anime['title']}', Reason: {anime['reason']}")
    
    results = {
        'Precision@K': safe_mean(precision_at_k),
        'NDCG': safe_mean(ndcg_scores),
        'Successfully Processed': processed_count,
        'Sample Size': len(sample_indices),
        'Skipped Anime': skipped_anime  # Include details in results
    }
    
    logging.info(f"Ranking quality evaluation completed: {results}")
    return results

def evaluate_with_user_ratings(num_users=1000):
    """
    Evaluate recommendations against actual user ratings
    
    Parameters:
    -----------
    num_users : int, optional
        Number of users to sample for evaluation
    
    Returns:
    --------
    dict
        Dictionary of user-based ranking metrics
    """
    logging.info(f"Starting user-based evaluation with {num_users} users")
    
    # Skip if ratings dataframe is empty
    if len(ratings_df) == 0:
        logging.warning("Skipping user-based evaluation - no ratings data available")
        return {'MAP': None, 'MRR': None, 'Users Processed': 0}
    
    # Get users who have rated multiple anime
    user_counts = ratings_df['user_id'].value_counts()
    users_with_multiple_ratings = user_counts[user_counts > 10].index.tolist()
    
    if not users_with_multiple_ratings:
        logging.warning("No users with multiple ratings found")
        return {'MAP': None, 'MRR': None, 'Users Processed': 0}
    
    # Sample users
    sample_users = np.random.choice(
        users_with_multiple_ratings, 
        min(num_users, len(users_with_multiple_ratings)), 
        replace=False
    )
    
    map_scores = []  # Mean Average Precision
    mrr_scores = []  # Mean Reciprocal Rank
    users_processed = 0
    
    for user_id in sample_users:
        try:
            # Get the user's ratings
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            
            # Remove -1 ratings (not rated)
            user_ratings = user_ratings[user_ratings['rating'] > 0]
            
            if len(user_ratings) < 5:  # Skip users with too few ratings
                continue
            
            # Split into "training" and "testing" sets (70%/30%)
            user_ratings = user_ratings.sample(frac=1).reset_index(drop=True)  # Shuffle
            split_idx = int(len(user_ratings) * 0.7)
            train_ratings = user_ratings.iloc[:split_idx]
            test_ratings = user_ratings.iloc[split_idx:]
            
            if len(train_ratings) == 0 or len(test_ratings) == 0:
                continue
            
            # Find a highly rated anime from the training set to use as input
            train_ratings = train_ratings.sort_values('rating', ascending=False)
            input_anime_id = train_ratings.iloc[0]['anime_id']
            
            # Map anime_id to name
            input_anime_rows = anime_df[anime_df['anime_id'] == input_anime_id]
            
            if len(input_anime_rows) == 0:
                continue
                
            input_anime_name = input_anime_rows.iloc[0]['name']
            
            # Get recommendations based on this anime
            recommendations = recommend_anime(input_anime_name, top_n=10)
            
            if isinstance(recommendations, str) or len(recommendations) == 0:
                continue
            
            # Get anime_ids from the recommendations
            rec_anime_ids = []
            for _, row in recommendations.iterrows():
                rec_name = row['name']
                rec_rows = anime_df[anime_df['name'] == rec_name]
                if len(rec_rows) > 0:
                    rec_anime_ids.append(rec_rows.iloc[0]['anime_id'])
            
            # Get the anime_ids that the user liked in the test set (rated > 7)
            liked_anime_ids = test_ratings[test_ratings['rating'] > 7]['anime_id'].values
            
            if len(liked_anime_ids) == 0:
                continue
            
            # Calculate Average Precision
            relevant_items = 0
            sum_precisions = 0
            
            for k, anime_id in enumerate(rec_anime_ids, 1):
                if anime_id in liked_anime_ids:
                    relevant_items += 1
                    precision_at_k = relevant_items / k
                    sum_precisions += precision_at_k
            
            ap = sum_precisions / len(liked_anime_ids) if len(liked_anime_ids) > 0 else 0
            map_scores.append(ap)
            
            # Calculate Reciprocal Rank (position of first relevant item)
            rr = 0
            for k, anime_id in enumerate(rec_anime_ids, 1):
                if anime_id in liked_anime_ids:
                    rr = 1 / k
                    break
            
            mrr_scores.append(rr)
            users_processed += 1
            logging.debug(f"Successfully processed user {users_processed}/{num_users}: {user_id}")
            
        except Exception as e:
            logging.error(f"Error processing user {user_id}: {e}")
            continue
    
    results = {
        'MAP': safe_mean(map_scores),
        'MRR': safe_mean(mrr_scores),
        'Users Processed': users_processed
    }
    
    logging.info(f"User-based evaluation completed: {results}")
    return results
    
def evaluate_coverage_diversity(top_n=5, sample_size=1000):
    """
    Evaluate what percentage of anime get recommended and how diverse the recommendations are
    """
    logging.info(f"Starting coverage and diversity evaluation with sample size: {sample_size}")
    
    # Sample anime to use as input
    sample_indices = np.random.choice(range(len(anime_df)), min(sample_size, len(anime_df)), replace=False)
    
    # Track all recommended anime
    all_recommended = set()
    diversity_scores = []
    
    for count, idx in enumerate(sample_indices):
        if count % 10 == 0:
            logging.debug(f"Processing diversity sample {count}/{len(sample_indices)}")
            
        try:
            title = anime_df.iloc[idx]['name']
            
            # Get recommendations
            recommendations = recommend_anime(title, top_n)
            
            if isinstance(recommendations, str):  # Error message
                continue
                
            if len(recommendations) == 0:
                continue
                
            # Add to set of all recommended anime
            all_recommended.update(recommendations.index.tolist())
            
            # Calculate diversity of this recommendation set
            if len(recommendations) > 1:
                genres_list = []
                for _, row in recommendations.iterrows():
                    try:
                        genre_str = row['genre']
                        if pd.notna(genre_str) and isinstance(genre_str, str) and genre_str != '':
                            genres = set(genre_str.split(', '))
                            genres = {g for g in genres if g}
                            if genres:
                                genres_list.append(genres)
                    except Exception as e:
                        logging.error(f"Error processing genre in diversity calculation: {e}")
                        continue
                
                # Calculate pairwise Jaccard distances
                if len(genres_list) > 1:
                    distances = []
                    for i in range(len(genres_list)):
                        for j in range(i+1, len(genres_list)):
                            try:
                                if not genres_list[i] or not genres_list[j]:
                                    continue
                                    
                                intersection = len(genres_list[i].intersection(genres_list[j]))
                                union = len(genres_list[i].union(genres_list[j]))
                                similarity = intersection / union if union > 0 else 0
                                distance = 1 - similarity  # Jaccard distance
                                distances.append(distance)
                            except Exception as e:
                                logging.error(f"Error calculating distance between genre sets: {e}")
                                continue
                    
                    if distances:
                        diversity_scores.append(np.mean(distances))
        except Exception as e:
            logging.error(f"Error in coverage evaluation for '{title}' (index {idx}): {e}")
            continue
    
    # Calculate coverage
    coverage = len(all_recommended) / len(anime_df) if len(anime_df) > 0 else 0
    
    results = {
        'Coverage': coverage,
        'Average Diversity': safe_mean(diversity_scores),
        'Unique Anime Recommended': len(all_recommended),
        'Total Anime': len(anime_df),
        'Sample Size': len([i for i in sample_indices if i < len(anime_df)])
    }
    
    logging.info(f"Coverage and diversity evaluation completed: {results}")
    return results

def evaluate_with_ratings(sample_size=100, similarity_threshold=0.8, max_pairs=200):
    """
    Validate if genre similarity aligns with user rating patterns - With detailed logging
    
    Parameters:
    -----------
    sample_size : int
        Maximum number of anime pairs to evaluate (default=100)
    similarity_threshold : float
        Only consider anime pairs with similarity above this threshold (default=0.8)
    max_pairs : int
        Maximum number of anime pairs to check before sampling (default=200)
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    logging.info(f"Starting rating validation with sample size: {sample_size}")
    
    # Skip if ratings dataframe is empty
    if len(ratings_df) == 0:
        logging.warning("Skipping ratings validation - no ratings data available")
        return {
            'Correlation': None,
            'Average Rating Difference': None,
            'Sample Size': 0
        }
    
    # Instead of checking all possible pairs (which is O(n²)), 
    # let's sample a subset of anime first
    anime_sample_size = min(1000, len(anime_df))
    anime_sample = np.random.choice(range(len(anime_df)), anime_sample_size, replace=False)
    
    logging.info(f"Finding similar anime pairs from {anime_sample_size} sampled anime...")
    similar_pairs = []
    
    # Track filtering metrics
    pairs_checked = 0
    filtered_out_reasons = {
        'low_similarity': 0,
        'max_pairs_reached': False,
        'max_checks_reached': False
    }
    
    # Find similar pairs only among the sample
    for i_idx in range(len(anime_sample)):
        i = anime_sample[i_idx]
        
        # Early termination if we've found enough pairs
        if len(similar_pairs) >= max_pairs:
            filtered_out_reasons['max_pairs_reached'] = True
            break
            
        # Process in batches and report progress
        if i_idx % 100 == 0 and i_idx > 0:
            logging.info(f"Processed {i_idx}/{anime_sample_size} anime, found {len(similar_pairs)} similar pairs")
        
        for j_idx in range(i_idx + 1, len(anime_sample)):
            j = anime_sample[j_idx]
            
            pairs_checked += 1
            # Skip if we've checked too many pairs
            if pairs_checked > 10000:  # Limit total pair checks
                filtered_out_reasons['max_checks_reached'] = True
                break
                
            # Safely check similarity
            if i < len(cosine_sim) and j < len(cosine_sim[i]):
                similarity = cosine_sim[i, j]
                if similarity > similarity_threshold:  # Higher threshold for efficiency
                    similar_pairs.append((i, j, similarity))
                else:
                    filtered_out_reasons['low_similarity'] += 1
                    
                # Early termination if we've found enough pairs
                if len(similar_pairs) >= max_pairs:
                    filtered_out_reasons['max_pairs_reached'] = True
                    break
    
    logging.info(f"Found {len(similar_pairs)} similar anime pairs after checking {pairs_checked} pairs")
    logging.info(f"Filtering summary: {filtered_out_reasons['low_similarity']} pairs below similarity threshold ({similarity_threshold})")
    if filtered_out_reasons['max_pairs_reached']:
        logging.info(f"Stopped searching after finding {max_pairs} pairs")
    if filtered_out_reasons['max_checks_reached']:
        logging.info(f"Stopped searching after checking 10000 pairs")
    
    # Take a sample if we have more than we need
    # Use random.sample instead of np.random.choice for lists of tuples
    if len(similar_pairs) > sample_size:
        import random
        similar_pairs = random.sample(similar_pairs, sample_size)
    
    # Map indices to anime_ids
    similar_pairs_ids = []
    indices_to_ids_errors = 0
    for i, j, sim in similar_pairs:
        # Make sure we have valid anime_ids
        if 'anime_id' in anime_df.columns and i < len(anime_df) and j < len(anime_df):
            try:
                anime1_id = anime_df.iloc[i]['anime_id']
                anime2_id = anime_df.iloc[j]['anime_id']
                similar_pairs_ids.append((anime1_id, anime2_id, sim))
            except Exception as e:
                logging.error(f"Error getting anime IDs for pair ({i}, {j}): {e}")
                indices_to_ids_errors += 1
                continue
                
    if indices_to_ids_errors > 0:
        logging.warning(f"Failed to map {indices_to_ids_errors} anime index pairs to anime IDs")
    
    # Get all anime IDs involved in our pairs
    all_anime_ids = set()
    for anime1_id, anime2_id, _ in similar_pairs_ids:
        all_anime_ids.add(anime1_id)
        all_anime_ids.add(anime2_id)
    
    logging.info(f"Processing ratings for {len(all_anime_ids)} unique anime")
    
    # Filter ratings to only those for our anime of interest
    relevant_ratings = ratings_df[ratings_df['anime_id'].isin(all_anime_ids)].copy()
    
    # Skip users with too few ratings - they won't have many anime pairs
    user_rating_counts = relevant_ratings['user_id'].value_counts()
    users_with_multiple = user_rating_counts[user_rating_counts > 1].index
    relevant_ratings = relevant_ratings[relevant_ratings['user_id'].isin(users_with_multiple)]
    
    # Skip -1 ratings (not rated)
    relevant_ratings = relevant_ratings[relevant_ratings['rating'] > 0]
    
    logging.info(f"Filtered ratings dataset: {len(relevant_ratings)} ratings from {len(users_with_multiple)} users")
    
    # Prepare for comparison
    rating_diffs = []
    pair_similarities = []
    pairs_without_common_raters = []
    
    logging.info(f"Comparing ratings for {len(similar_pairs_ids)} anime pairs...")
    
    # Process each pair
    for count, (anime1_id, anime2_id, similarity) in enumerate(similar_pairs_ids):
        if count % 10 == 0 and count > 0:
            logging.info(f"Processed {count}/{len(similar_pairs_ids)} anime pairs")
            
        try:
            # Get anime names for logging
            anime1_name = anime_df[anime_df['anime_id'] == anime1_id]['name'].values[0] if len(anime_df[anime_df['anime_id'] == anime1_id]) > 0 else f"ID:{anime1_id}"
            anime2_name = anime_df[anime_df['anime_id'] == anime2_id]['name'].values[0] if len(anime_df[anime_df['anime_id'] == anime2_id]) > 0 else f"ID:{anime2_id}"
            
            # Get ratings for each anime
            ratings1 = relevant_ratings[relevant_ratings['anime_id'] == anime1_id]
            ratings2 = relevant_ratings[relevant_ratings['anime_id'] == anime2_id]
            
            # Find common users efficiently using merge
            common_ratings = pd.merge(
                ratings1, 
                ratings2, 
                on='user_id', 
                suffixes=('_1', '_2')
            )
            
            # If no common raters, track this pair
            if len(common_ratings) == 0:
                pairs_without_common_raters.append({
                    'anime1_id': anime1_id,
                    'anime1_name': anime1_name,
                    'anime2_id': anime2_id,
                    'anime2_name': anime2_name,
                    'similarity': similarity
                })
                continue
                
            # Calculate differences
            for _, row in common_ratings.iterrows():
                rating1 = row['rating_1']
                rating2 = row['rating_2']
                
                rating_diff = abs(rating1 - rating2)
                rating_diffs.append(rating_diff)
                pair_similarities.append(similarity)
                
                # Early termination if we have enough data points
                if len(rating_diffs) >= 10000:
                    logging.info("Reached maximum number of rating comparisons")
                    break
                    
            if len(rating_diffs) >= 10000:
                break
                
        except Exception as e:
            logging.error(f"Error processing ratings for anime pair ({anime1_id}, {anime2_id}): {e}")
            continue
    
    # Log information about pairs without common raters
    if pairs_without_common_raters:
        logging.info(f"{len(pairs_without_common_raters)} anime pairs had no common raters:")
        for i, pair in enumerate(pairs_without_common_raters[:10], 1):  # Show first 10 examples
            logging.info(f"  {i}. '{pair['anime1_name']}' and '{pair['anime2_name']}' (similarity: {pair['similarity']:.4f})")
        if len(pairs_without_common_raters) > 10:
            logging.info(f"  ... and {len(pairs_without_common_raters) - 10} more pairs")
    
    # Calculate correlation between similarity and rating difference
    correlation = safe_corrcoef(pair_similarities, rating_diffs)
    # Note: we expect a negative correlation (higher similarity -> lower rating difference)
    
    results = {
        'Correlation': correlation,
        'Average Rating Difference': safe_mean(rating_diffs),
        'Sample Size': len(rating_diffs),
        'Pairs Without Common Raters': len(pairs_without_common_raters),
        'Total Pairs Checked': pairs_checked
    }
    
    logging.info(f"Rating validation completed: {results}")
    return results
    
# Updated main evaluation function to incorporate the new metrics
def evaluate_recommender_system():
    """Main evaluation function that runs all metrics and collects results"""
    logging.info("Starting evaluation of anime recommender system...")
    
    # 1. Evaluate similarity quality
    logging.info("Evaluating similarity metrics...")
    similarity_results = evaluate_similarity_quality()
    
    # 2. Evaluate ranking quality
    logging.info("Evaluating ranking metrics...")
    ranking_results = evaluate_ranking_quality(num_samples=1000)
    
    # 3. Evaluate coverage and diversity
    logging.info("Evaluating coverage and diversity...")
    coverage_results = evaluate_coverage_diversity()
    
    # 4. External validation with ratings (if available)
    logging.info("Performing external validation with ratings...")
    if len(ratings_df) > 0:
        # Original ratings validation
        ratings_validation = evaluate_with_ratings()
        
        # New user-based evaluation
        logging.info("Performing user-based evaluation...")
        user_results = evaluate_with_user_ratings(num_users=1000)
        
        # Combine results
        ratings_validation.update(user_results)
    else:
        logging.warning("Skipping ratings validation due to missing ratings data")
        ratings_validation = {
            'Correlation': None,
            'Average Rating Difference': None,
            'Sample Size': 0,
            'MAP': None,
            'MRR': None,
            'Users Processed': 0
        }
    
    # Compile all results
    all_results = {
        "Similarity Metrics": similarity_results,
        "Ranking Metrics": ranking_results,
        "Coverage and Diversity": coverage_results,
        "External Validation": ratings_validation
    }
    
    # Log all results
    logging.info("\nEvaluation Results:")
    for category, results in all_results.items():
        logging.info(f"\n{category}:")
        for metric, value in results.items():
            logging.info(f"  {metric}: {value}")
    
    return all_results

# Added function to save results to CSV
def save_results_to_csv(results, filename=None):
    """
    Save evaluation results to a CSV file
    
    Parameters:
    -----------
    results : dict
        Dictionary of evaluation results
    filename : str, optional
        Name of the CSV file (default: recommender_results_{timestamp}.csv)
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recommender_results_{timestamp}.csv"
    
    # Flatten the nested dictionary
    flat_results = {}
    for category, metrics in results.items():
        for metric, value in metrics.items():
            flat_results[f"{category}_{metric}"] = value
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame([flat_results])
    results_df.to_csv(filename, index=False)
    logging.info(f"Results saved to {filename}")
    return filename

if __name__ == "__main__":
    try:
        logging.info("Starting anime recommender system evaluation")
        results = evaluate_recommender_system()
        
        # Save results to CSV
        csv_file = save_results_to_csv(results)
        
        logging.info(f"\nEvaluation completed successfully! Results saved to {csv_file}")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        import traceback
        logging.error(traceback.format_exc())