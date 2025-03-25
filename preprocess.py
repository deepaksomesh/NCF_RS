import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============
# 1. Load the MovieLens dataset
# ============
# We'll read the ratings.dat file from the MovieLens 1M dataset.
# The file has the format userId::movieId::rating::timestamp, and we’ll keep all columns for now.

def load_data(file_path):
    # Read the file using pandas with :: as the separator and assign column names
    data = pd.read_csv(file_path, sep='::', header=None, 
                       names=['userId', 'movieId', 'rating', 'timestamp'], 
                       engine='python')
    return data

# ============
# 2. Convert ratings to binary implicit feedback
# ============
# User ratings >= 4 as positive interactions as label = 1 else label = 0.

def convert_to_implicit(data):
    data['label'] = np.where(data['rating'] >= 4, 1, 0)
    positive_data = data[data['label'] == 1][['userId', 'movieId', 'rating', 'timestamp', 'label']]
    return positive_data

# ============
# 3. Perform negative sampling
# ============
# We need negative samples (label = 0) for movies users haven’t rated. For each user, we’ll sample 4 non-interacted movies.

def negative_sampling(positive_data, num_negatives_per_positive=4):
    # Get unique users and movies from the positive data
    all_users = positive_data['userId'].unique()
    all_movies = positive_data['movieId'].unique()
    
    # Create a set of (user, movie) pairs from positive data for quick lookup
    interacted_pairs = set(zip(positive_data['userId'], positive_data['movieId']))
    
    negative_samples = []
    
    for user in all_users:
        # Movies this user has rated positively
        user_movies = positive_data[positive_data['userId'] == user]['movieId'].values
        # Movies this user hasn’t interacted with
        non_interacted = np.setdiff1d(all_movies, user_movies)
        
        # Calculate number of negatives 
        num_positives = len(user_movies)
        num_negatives = min(num_positives * num_negatives_per_positive, len(non_interacted))
        
        # Randomly sample negative movies
        sampled_negatives = np.random.choice(non_interacted, size=num_negatives, replace=False)
        
        # For each negative movie, add a dummy timestamp
        min_timestamp = positive_data['timestamp'].min()
        for movie in sampled_negatives:
            negative_samples.append([user, movie, 0, min_timestamp, 0])
    

    negative_df = pd.DataFrame(negative_samples, columns=['userId', 'movieId', 'rating', 'timestamp', 'label'])
    
    # Combine positive and negative data
    full_data = pd.concat([positive_data, negative_df], ignore_index=True)
    return full_data

# ============
# 4. Split data into 70% training, 15% validation, and 15% test sets
# ============

def split_data(data):
    # First split: 70% train, 30% remaining
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    
    # Second split: 15% validation, 15% test from the remaining 30%
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    return train_data, val_data, test_data

# ============
# 5. Main function
# ============

def main():
    print("** Loading raw rating data from Movielens **") 
    file_path = "ratings.dat"
    raw_data = load_data(file_path)
    print("Raw data loaded:", raw_data.shape)

    # Postive sampling
    positive_data = convert_to_implicit(raw_data)
    print("Positive interactions:", positive_data.shape)
    
    # Negative sampling
    full_data = negative_sampling(positive_data)
    print("After negative sampling:", full_data.shape)
    
    # Split the datasets 
    train_data, val_data, test_data = split_data(full_data)
    print("Train set:", train_data.shape)
    print("Validation set:", val_data.shape)
    print("Test set:", test_data.shape)
    
    train_data.to_csv("train_data.csv", index=False)
    val_data.to_csv("val_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)
    print("Data preprocessing complete! Files saved as train_data.csv, val_data.csv, and test_data.csv.")

# ============
# 6. Execute the main function
# ============

if __name__ == "__main__":
    main()