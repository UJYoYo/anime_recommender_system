# Anime Recommendation System

A content-based recommendation system for anime built using TF-IDF vectorization and cosine similarity on genre data. This project includes both a web-based demo interface and comprehensive evaluation metrics.

## Overview

This project uses data from the [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) on Kaggle, which contains information on user preference data from 73,516 users on 12,294 anime. The recommendation engine works by:

1. Computing text-based similarities between anime genres using TF-IDF vectorization
2. Filtering recommendations by anime type (TV, Movie, OVA, etc.)
3. Ranking the results by similarity score and user ratings

## Project Structure

```
├── recommender.ipynb       # Main Jupyter notebook with algorithm development and evaluation
├── results/                # Directory containing evaluation results
│   ├── evaluation_results.csv
│   └── full_evaluation_result.log
└── web_demo/               # Web-based demonstration of the recommendation system
    ├── app.py              # Flask backend for recommendation API
    ├── index.html          # Frontend interface with dynamic recommendations
    └── style.css           # Styling for the web interface
```

## Features

- **Content-based recommendations** based on genre similarity
- **Type-aware filtering** to suggest similar content types (TV shows with TV shows, movies with movies)
- **Interactive web interface** with search functionality and detailed anime information
- **Real-time recommendations** via Flask API backend
- **Comprehensive evaluation metrics** for recommendation quality

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, flask, flask-cors

### Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/anime-recommender.git
cd anime-recommender
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the dataset
```python
import kagglehub
path = kagglehub.dataset_download("CooperUnion/anime-recommendations-database")
```

### Running the Web Demo

1. Start the Flask server
```bash
cd web_demo
python app.py
```

2. Open `index.html` in your browser or serve it with a simple HTTP server
```bash
python -m http.server
```

3. Access the demo at http://localhost:8000/index.html

## Algorithm Details

The recommendation system uses the following approach:

1. **Preprocessing**: Convert anime genres to TF-IDF vectors
2. **Similarity Calculation**: Compute cosine similarity between anime genre vectors
3. **Filtering**: Filter recommendations by anime type
4. **Ranking**: Sort by similarity score and rating

## Evaluation Metrics

The system was evaluated using several metrics:

- **Precision@K**: 0.999 - How many of the recommended items are relevant
- **NDCG**: 0.955 - Normalized Discounted Cumulative Gain
- **Coverage**: 0.247 - What percentage of items can be recommended
- **Diversity**: 0.191 - How different recommendations are from each other
- **MRR (Mean Reciprocal Rank)**: 0.114 - Position of first relevant item
- **MAP (Mean Average Precision)**: 0.013 - Precision across all relevant items

## Web Interface

The web interface provides:
- Browsing of top-rated anime
- Filtering by genre
- Sorting by rating, alphabetically, or by episode count
- Detailed anime information
- Similar anime recommendations for each title
- Recommendation fallback for offline use

## Future Improvements

- Incorporate collaborative filtering to provide personalized recommendations
- Add more metadata for filtering (year, studio, staff)
- Implement user accounts to save favorite anime
- Expand the dataset with more recent anime titles
- Improve recommendation diversity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data provided by [Cooper Union](https://www.kaggle.com/CooperUnion/anime-recommendations-database) on Kaggle
- Based on MyAnimeList.net data
