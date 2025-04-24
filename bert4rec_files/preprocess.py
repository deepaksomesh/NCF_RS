import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os # For checking file existence
import config

# Step 2: Define function to load and preprocess data for BERT4Rec
def load_and_preprocess_data_bert4rec(max_len=config.MAX_LEN):
    """
    Loads the MovieLens 1M dataset, filters interactions (rating >= 4),
    processes it into user interaction sequences, maps item IDs, adds special tokens,
    pads/truncates sequences, filters users (<5 interactions), and splits users
    into train/val/test sets. Conforms to requirements.

    Returns:
        train_sequences (dict): User ID -> sequence for training.
        val_sequences (dict): User ID -> (sequence, target_item) for validation.
        test_sequences (dict): User ID -> (sequence, target_item) for testing.
        user_map (dict): Original userId -> contiguous integer ID.
        item_map (dict): Original movieId -> contiguous integer ID (starting from 1).
        num_users (int): Total number of unique users after filtering.
        num_items_with_special_tokens (int): Total number of unique items + special tokens.
        item_rev_map (dict): Contiguous item ID -> Original movieId
        pad_token_id (int): ID for the padding token.
        mask_token_id (int): ID for the mask token.
        user_sequences_full (dict): Mapped User ID -> Full list of mapped item IDs (for negative sampling).
    """
    # ... (previous loading and filtering code remains the same) ...
    print("Loading MovieLens 1M dataset...")
    ratings_file = "ratings.dat"
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"{ratings_file} not found. Please download MovieLens 1M dataset.")

    ratings = pd.read_csv(ratings_file, sep="::", engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
    print(f"Original interactions: {len(ratings)}")
    ratings = ratings[ratings['rating'] >= 4].copy()
    print(f"Interactions after filtering (rating >= 4): {len(ratings)}")

    print("Filtering out users/items with few interactions (threshold >= 5)...")
    while True:
        user_counts = ratings['userId'].value_counts()
        item_counts = ratings['movieId'].value_counts()
        initial_users = len(user_counts)
        initial_items = len(item_counts)
        valid_users = user_counts[user_counts >= 5].index
        ratings = ratings[ratings['userId'].isin(valid_users)]
        valid_items = ratings['movieId'].value_counts()[lambda x: x >= 5].index
        ratings = ratings[ratings['movieId'].isin(valid_items)]
        valid_users = ratings['userId'].value_counts()[lambda x: x >= 5].index
        ratings = ratings[ratings['userId'].isin(valid_users)]
        final_users = len(valid_users)
        # Need to update final_items count based on the last filtering step
        final_items = len(ratings['movieId'].unique())  # More accurate count after filtering

        print(f"Filtering loop: Users {initial_users}->{final_users}, Items {initial_items}->{final_items}")
        if initial_users == final_users and initial_items == len(valid_items):  # Check item count stability too
            # Need a stable check based on actual valid items remaining
            current_valid_items = ratings['movieId'].value_counts()[lambda x: x >= 5].index
            if initial_users == final_users and len(valid_items) == len(current_valid_items):
                break  # Stop if no users/items were removed in this iteration

    print(f"Interactions after filtering users/items (<5 interactions): {len(ratings)}")

    print("Mapping user and item IDs...")
    unique_users = sorted(ratings['userId'].unique())
    unique_items = sorted(ratings['movieId'].unique())
    user_map = {user_id: i for i, user_id in enumerate(unique_users)}
    item_map = {item_id: i + 1 for i, item_id in enumerate(unique_items)}
    item_rev_map = {v: k for k, v in item_map.items()}
    num_users = len(user_map)
    num_items = len(item_map)  # Actual number of items
    pad_token_id = 0
    mask_token_id = num_items + 1
    num_items_with_special_tokens = num_items + 2
    print(f"Num users after filtering: {num_users}, Num items after filtering: {num_items}")
    print(f"Num items including special tokens: {num_items_with_special_tokens}")
    print(f"PAD ID: {pad_token_id}, MASK ID: {mask_token_id}")

    ratings['userId_mapped'] = ratings['userId'].map(user_map)
    ratings['movieId_mapped'] = ratings['movieId'].map(item_map)

    print("Sorting interactions by user and timestamp...")
    ratings = ratings.sort_values(['userId_mapped', 'timestamp'])

    print("Grouping interactions into sequences...")
    # Store the full sequence with *mapped* item IDs using the *mapped* user ID as key
    user_sequences_full = ratings.groupby('userId_mapped')['movieId_mapped'].apply(list).to_dict()

    # --- User Splitting (remains the same) ---
    all_user_ids = list(user_sequences_full.keys())
    train_user_ids, temp_user_ids = train_test_split(all_user_ids, test_size=0.3, random_state=42)
    val_user_ids, test_user_ids = train_test_split(temp_user_ids, test_size=0.5, random_state=42)
    print(f"Splitting users: {len(train_user_ids)} train, {len(val_user_ids)} val, {len(test_user_ids)} test")

    # --- Sequence Creation using Leave-One-Out (remains the same) ---
    train_sequences_raw = {}
    val_sequences_raw = {}
    test_sequences_raw = {}
    print("Applying leave-one-out split logic for val/test...")
    for user_id in tqdm(train_user_ids, desc="Processing train users"):
        seq = user_sequences_full.get(user_id, [])
        if len(seq) >= 2:
            train_sequences_raw[user_id] = seq[:-1][-max_len:]
    for user_id in tqdm(val_user_ids, desc="Processing val users"):
        seq = user_sequences_full.get(user_id, [])
        if len(seq) >= 2:
            val_sequences_raw[user_id] = (seq[:-1][-max_len:], seq[-1])
    for user_id in tqdm(test_user_ids, desc="Processing test users"):
        seq = user_sequences_full.get(user_id, [])
        if len(seq) >= 2:
            test_sequences_raw[user_id] = (seq[-max_len:], seq[-1])

    train_sequences = {k: v for k, v in train_sequences_raw.items() if len(v) > 0}
    val_sequences = {k: v for k, v in val_sequences_raw.items() if len(v[0]) > 0}
    test_sequences = {k: v for k, v in test_sequences_raw.items() if len(v[0]) > 0}
    print(
        f"Final sequences after filtering empty: {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test")

    # Also convert user_sequences_full to use sets for faster lookups during negative sampling
    user_sequences_full_sets = {user_id: set(seq) for user_id, seq in user_sequences_full.items()}

    return (train_sequences, val_sequences, test_sequences,
            user_map, item_map, item_rev_map,
            num_users, num_items_with_special_tokens,
            pad_token_id, mask_token_id,
            user_sequences_full_sets)  # Return the full history sets