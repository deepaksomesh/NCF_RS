
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import random
import os

# Step 1: Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Constants
MAX_LEN = 200
BERT_DROPOUT = 0.2
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 0.0003
EPOCHS = 200
PATIENCE = 10
TOP_K = 10
WEIGHT_DECAY = 0.01
NUM_FUTURE_ITEMS = 5

# Step 2: Load and preprocess data
def load_and_preprocess_data_bert4rec(max_len=MAX_LEN, num_future_items=NUM_FUTURE_ITEMS): # Added num_future_items param
    """
    Loads MovieLens 1M, filters, maps IDs, and creates sequences.
    Implements multi-item chronological splitting within each user's sequence after the initial user train/val/test split.
    - Train users: Use sequence up to the last `num_future_items` interactions for MLM training.
    - Val/Test users: Use sequence up to the last `num_future_items` interactions as input,
                      predict the list of the last `num_future_items` interactions.
    """
    print("Loading MovieLens 1M dataset...")
    ratings_file = "ratings.dat"
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"{ratings_file} not found.")

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
        # Filter users with less than 5 interactions
        valid_users = user_counts[user_counts >= 5].index
        ratings = ratings[ratings['userId'].isin(valid_users)]
        # Filter items with less than 5 interactions based on remaining users
        item_counts_after_user_filter = ratings['movieId'].value_counts()
        valid_items = item_counts_after_user_filter[item_counts_after_user_filter >= 5].index
        ratings = ratings[ratings['movieId'].isin(valid_items)]
        # Re-filter users to ensure they still have >= 5 interactions after item filtering
        user_counts_after_item_filter = ratings['userId'].value_counts()
        valid_users = user_counts_after_item_filter[user_counts_after_item_filter >= 5].index
        ratings = ratings[ratings['userId'].isin(valid_users)]

        final_users = len(ratings['userId'].unique())
        final_items = len(ratings['movieId'].unique()) # Current count of unique items
        print(f"Filtering loop: Users {initial_users}->{final_users}, Items {initial_items}->{final_items}")

        # Check if the number of users and valid items stabilized
        current_item_counts = ratings['movieId'].value_counts()
        current_valid_items = current_item_counts[current_item_counts >= 5].index
        if initial_users == final_users and len(valid_items) == len(current_valid_items):
                break
        else:
             valid_items = current_valid_items


    print(f"Interactions after filtering users/items (<5 interactions): {len(ratings)}")

    print("Mapping user and item IDs...")
    unique_users = sorted(ratings['userId'].unique())
    unique_items = sorted(ratings['movieId'].unique())
    user_map = {user_id: i for i, user_id in enumerate(unique_users)}
    item_map = {item_id: i + 1 for i, item_id in enumerate(unique_items)}
    item_rev_map = {v: k for k, v in item_map.items()}
    num_users = len(user_map)
    num_items = len(item_map)
    pad_token_id = 0
    mask_token_id = num_items + 1
    num_items_with_special_tokens = num_items + 2
    print(f"Num users: {num_users}, Num items: {num_items}")
    print(f"Num items including special tokens (PAD={pad_token_id}, MASK={mask_token_id}): {num_items_with_special_tokens}")
    print(f"Using Future Items = {num_future_items} for evaluation targets.")

    ratings['userId_mapped'] = ratings['userId'].map(user_map)
    ratings['movieId_mapped'] = ratings['movieId'].map(item_map)

    print("Sorting interactions by user and timestamp...")
    ratings = ratings.sort_values(['userId_mapped', 'timestamp'])

    print("Grouping interactions into sequences...")
    user_sequences_full = ratings.groupby('userId_mapped')['movieId_mapped'].apply(list).to_dict()

    # --- User Splitting ---
    all_user_ids = list(user_sequences_full.keys())
    train_user_ids, temp_user_ids = train_test_split(all_user_ids, test_size=0.3, random_state=42)
    val_user_ids, test_user_ids = train_test_split(temp_user_ids, test_size=0.5, random_state=42)
    print(f"Splitting users: {len(train_user_ids)} train, {len(val_user_ids)} val, {len(test_user_ids)} test")
    train_sequences_raw = {}
    val_sequences_raw = {}
    test_sequences_raw = {}
    print("Applying chronological split...")

    # Training sequences: Use interactions up to the last `num_future_items` for MLM
    for user_id in tqdm(train_user_ids, desc="Processing train users"):
        seq = user_sequences_full.get(user_id, [])
        # Need at least num_future_items + 1 interaction for training history
        if len(seq) >= num_future_items + 1:
            train_sequences_raw[user_id] = seq[:-num_future_items][-max_len:]

    # Validation sequences: Predict the last `num_future_items` using interactions before them
    for user_id in tqdm(val_user_ids, desc="Processing val users"):
        seq = user_sequences_full.get(user_id, [])
        if len(seq) >= num_future_items + 1:
            input_seq = seq[:-num_future_items][-max_len:]
            target_items = seq[-num_future_items:]
            if len(target_items) == num_future_items:
                val_sequences_raw[user_id] = (input_seq, target_items)

    # Test sequences
    for user_id in tqdm(test_user_ids, desc="Processing test users"):
        seq = user_sequences_full.get(user_id, [])
        # Need at least num_future_items + 1 interactions
        if len(seq) >= num_future_items + 1:
            input_seq = seq[:-num_future_items][-max_len:] # History before future items
            target_items = seq[-num_future_items:]         # List of last num_future_items
            if len(target_items) == num_future_items: # Ensure we got the right number
                test_sequences_raw[user_id] = (input_seq, target_items)

    # Filter out any empty input sequences
    train_sequences = {k: v for k, v in train_sequences_raw.items() if len(v) > 0}
    val_sequences = {k: v for k, v in val_sequences_raw.items() if len(v[0]) > 0}
    test_sequences = {k: v for k, v in test_sequences_raw.items() if len(v[0]) > 0}
    print(f"Final sequences after multi-chrono split & filtering empty:"
          f" {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test")

    # Create sets of all items interacted with by each user for negative sampling during evaluation
    user_sequences_full_sets = {user_id: set(seq) for user_id, seq in user_sequences_full.items()}

    return (train_sequences, val_sequences, test_sequences, user_map, item_map, item_rev_map,
            num_users, num_items_with_special_tokens, pad_token_id, mask_token_id, user_sequences_full_sets)


# Step 3: Dataset class
class BERT4RecDataset(Dataset):
    def __init__(self, user_sequences, max_len, mask_prob, mask_token_id, pad_token_id, num_items, mode='train', num_future_items=NUM_FUTURE_ITEMS): # Add num_future_items
        self.user_ids = list(user_sequences.keys())
        self.sequences_data = [user_sequences[uid] for uid in self.user_ids]
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.num_items = num_items
        self.mode = mode
        self.num_future_items = num_future_items

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        if self.mode == 'train':
            seq = self.sequences_data[idx]
        else:
            seq, target_items = self.sequences_data[idx]

        seq_len = len(seq)
        padding_len = max(0, self.max_len - seq_len)
        padded_seq = seq + [self.pad_token_id] * padding_len
        input_ids = np.array(padded_seq[:self.max_len])

        if self.mode == 'train':
            output_seq = input_ids.copy()
            labels = np.full(self.max_len, self.pad_token_id, dtype=np.int64)
            for i in range(seq_len):
                prob = random.random()
                if prob < self.mask_prob:
                    labels[i] = input_ids[i]
                    mask_decision = random.random()
                    if mask_decision < 0.8:
                        output_seq[i] = self.mask_token_id
                    elif mask_decision < 0.9:
                        random_item_id = random.randint(1, self.num_items)
                        output_seq[i] = random_item_id
            if seq_len > 0 and (labels == self.pad_token_id).all():
                mask_pos = random.randrange(seq_len)
                labels[mask_pos] = input_ids[mask_pos]
                output_seq[mask_pos] = self.mask_token_id
            return torch.tensor(output_seq, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
        # mode == 'val' or 'test'
        else:
            output_seq = input_ids.copy()
            if seq_len > 0:
                mask_pos = seq_len - 1
                output_seq[mask_pos] = self.mask_token_id
                return torch.tensor(output_seq, dtype=torch.long), torch.tensor(target_items, dtype=torch.long), torch.tensor(user_id, dtype=torch.long)
            else:
                 # Handle edge case of empty input sequence
                 # Return dummy tensors, including a target tensor of expected shape
                 dummy_targets = [self.pad_token_id] * self.num_future_items
                 return torch.tensor(output_seq, dtype=torch.long), torch.tensor(dummy_targets, dtype=torch.long), torch.tensor(-1, dtype=torch.long)


# Step 4: Model
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len = x.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.pe(position_ids)

class BERT4Rec(nn.Module):
    def __init__(self, num_items, hidden_dim, max_len, n_layers, n_heads, dropout, pad_token_id):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.mask_token_id = num_items - 1
        self.item_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=pad_token_id)
        self.positional_embedding = PositionalEmbedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_layernorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=False
        )
        encoder_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=encoder_norm)
        self.output_layer = nn.Linear(hidden_dim, num_items)
        self._init_weights()

    def _init_weights(self):
        stddev = 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if module.weight.requires_grad:
                    torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=stddev, a=-2*stddev, b=2*stddev)
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                if module.bias.requires_grad:
                    module.bias.data.zero_()
        if hasattr(self.positional_embedding, 'pe'):
             torch.nn.init.trunc_normal_(self.positional_embedding.pe.weight, mean=0.0, std=stddev, a=-2*stddev, b=2*stddev)
        if self.item_embedding.padding_idx is not None:
            with torch.no_grad():
                self.item_embedding.weight[self.item_embedding.padding_idx].fill_(0)

    def forward(self, input_ids):
        attention_mask = (input_ids == self.pad_token_id)
        item_emb = self.item_embedding(input_ids)
        pos_emb = self.positional_embedding(input_ids)
        embedding = item_emb + pos_emb
        embedding = self.emb_layernorm(embedding)
        embedding = self.emb_dropout(embedding)
        transformer_output = self.transformer_encoder(src=embedding, src_key_padding_mask=attention_mask)
        logits = self.output_layer(transformer_output)
        return logits


# Step 5: Training function
def train_model_bert4rec(model, train_loader, val_loader, user_sequences_full_sets, num_items,
                         num_items_with_special_tokens, pad_token_id, device, config,
                         epochs=EPOCHS, patience=PATIENCE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, top_k=TOP_K):
    model_name = config['name']
    print(f"\n=== Training {model_name} ===")
    print(f"Config: {config}")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.1, 1.0 - epoch / (epochs * 1.5)))

    best_val_ndcg = -1
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        train_progress = tqdm(train_loader, desc=f"Training {model_name}")
        for input_ids, labels in train_progress:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            train_progress.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")

        # Validation phase
        model.eval()
        val_recall, val_ndcg = evaluate_model_bert4rec( # Calls the multi-item eval function
            model, val_loader, user_sequences_full_sets, num_items, device, top_k, num_negatives=100
        )
        print(f"Val Recall@{top_k}: {val_recall:.4f}")
        print(f"Val NDCG@{top_k}: {val_ndcg:.4f}")

        # Early stopping logic
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            print(f"New best model found! Validation NDCG@{top_k}: {best_val_ndcg:.4f}\n")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation NDCG@{top_k}. Epochs without improvement: {epochs_no_improve}/{patience}\n")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state:
        print(f"Loading best model state with validation NDCG@{top_k}: {best_val_ndcg:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state saved. Using the state from the last epoch.")
    return model


# Step 6: Evaluation
def evaluate_model_bert4rec(model, data_loader, user_sequences_full_sets, num_items, device, top_k=TOP_K, num_negatives=100):
    """
    Evaluates the BERT4Rec model using Recall@K and NDCG@K for MULTIPLE future items.
    """
    model.eval()
    recall_list = []
    ndcg_list = []
    pad_token_id = model.pad_token_id
    mask_token_id = model.mask_token_id
    all_item_ids = set(range(1, num_items + 1))
    eps = 1e-9

    eval_progress = tqdm(data_loader, desc="Evaluating")
    with torch.no_grad():
        for input_ids, target_items_batch, user_ids_batch in eval_progress:
            input_ids = input_ids.to(device)
            target_items_batch = target_items_batch.to(device)
            user_ids_batch = user_ids_batch.to(device)

            logits = model(input_ids)

            batch_recalls = []
            batch_ndcgs = []

            for i in range(input_ids.size(0)):
                user_id = user_ids_batch[i].item()
                # Skip dummy users
                if user_id == -1:
                    continue

                seq_with_mask = input_ids[i]
                # Get the list of ground truth items for the user
                # Filter out any potential padding if dataset/dataloader is adding it
                true_target_items_list = [item_id for item_id in target_items_batch[i].tolist() if item_id != pad_token_id]
                if not true_target_items_list:
                    continue

                true_target_items_set = set(true_target_items_list)
                num_true_targets = len(true_target_items_set)

                # --- Prediction and Ranking ---
                mask_positions = (seq_with_mask == mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_positions) == 0:
                    actual_len = (seq_with_mask != pad_token_id).sum().item()
                    if actual_len > 0:
                        mask_pos = actual_len - 1
                    else:
                        continue
                else:
                    # Uses the last mask position
                    mask_pos = mask_positions[-1].item()

                all_scores = logits[i, mask_pos, :]

                # --- Negative Sampling ---
                historical_items = user_sequences_full_sets.get(user_id, set())
                # Exclude items the user interacted with historically AND the future target items
                items_to_exclude_sampling = historical_items | true_target_items_set
                possible_negatives = list(all_item_ids - items_to_exclude_sampling)

                if len(possible_negatives) < num_negatives:
                    negatives = random.choices(possible_negatives, k=num_negatives) if possible_negatives else []
                else:
                    negatives = random.sample(possible_negatives, num_negatives)

                # Candidate list: ground truth items + negative samples
                candidate_items = list(true_target_items_set) + negatives
                candidate_items = [item for item in candidate_items if 1 <= item <= num_items]
                if not candidate_items: continue

                candidate_items = sorted(list(set(candidate_items)))

                # Get scores only for the unique candidate items
                candidate_scores = all_scores[candidate_items]

                # Get top-K recommended items from the candidates
                _, top_k_indices_relative = torch.topk(candidate_scores, k=min(top_k, len(candidate_items)))
                top_k_items = [candidate_items[idx.item()] for idx in top_k_indices_relative]
                top_k_set = set(top_k_items)

                # --- Calculate Recall@K ---
                hits = top_k_set & true_target_items_set
                recall = len(hits) / (num_true_targets + eps)
                batch_recalls.append(recall)

                # --- Calculate NDCG@K ---
                dcg = 0.0
                for rank, item in enumerate(top_k_items):
                    if item in true_target_items_set:
                        dcg += 1.0 / math.log2(rank + 2) # rank is 0-based

                # Calculate Ideal DCG
                idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(num_true_targets, top_k)))

                ndcg = dcg / (idcg + eps)
                batch_ndcgs.append(ndcg)

            if batch_recalls: recall_list.extend(batch_recalls)
            if batch_ndcgs: ndcg_list.extend(batch_ndcgs)

    avg_recall = np.mean(recall_list) if recall_list else 0.0
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    return avg_recall, avg_ndcg


# Step 7: Main function
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print("Loading and preprocessing data...")
    try:
        (train_sequences, val_sequences, test_sequences, user_map, item_map, item_rev_map,
         num_users, num_items_with_special_tokens, pad_token_id, mask_token_id, user_sequences_full_sets
         ) = load_and_preprocess_data_bert4rec(max_len=MAX_LEN, num_future_items=NUM_FUTURE_ITEMS) # Pass here
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'ratings.dat' from the MovieLens 1M dataset is in the same directory.")
        return

    num_items = len(item_map) # Actual number of unique items
    print(f"Actual number of items (used for sampling/eval): {num_items}")
    print(f"Vocabulary size (for embedding/output layer): {num_items_with_special_tokens}")
    print(f"Number of users: {num_users}")

    # configurations
    model_configs = [
        {"name": f"BERT4Rec-M-Multi {NUM_FUTURE_ITEMS}", "hidden_dim": 128, "n_layers": 2, "n_heads": 2, "mask_prob": 0.2},
        {"name": f"BERT4Rec-S-Multi {NUM_FUTURE_ITEMS}", "hidden_dim": 64, "n_layers": 2, "n_heads": 2, "mask_prob": 0.2},
        {"name": f"BERT4Rec-L-Multi {NUM_FUTURE_ITEMS}", "hidden_dim": 256, "n_layers": 2, "n_heads": 2, "mask_prob": 0.2},
        {"name": f"BERT4Rec-XL-Multi {NUM_FUTURE_ITEMS}", "hidden_dim": 256, "n_layers": 4, "n_heads": 4, "mask_prob": 0.4}
    ]

    results = []
    num_workers = 0
    print(f"Using num_workers={num_workers} for DataLoaders.")

    for config in model_configs:
        print(f"\n--- Starting Configuration: {config['name']} ---")
        # Create datasets and dataloaders
        train_dataset = BERT4RecDataset(train_sequences, MAX_LEN, config["mask_prob"], mask_token_id, pad_token_id, num_items, mode='train', num_future_items=NUM_FUTURE_ITEMS)
        val_dataset = BERT4RecDataset(val_sequences, MAX_LEN, config["mask_prob"], mask_token_id, pad_token_id, num_items, mode='val', num_future_items=NUM_FUTURE_ITEMS)
        test_dataset = BERT4RecDataset(test_sequences, MAX_LEN, config["mask_prob"], mask_token_id, pad_token_id, num_items, mode='test', num_future_items=NUM_FUTURE_ITEMS)

        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=num_workers)

        # Initialize model
        model = BERT4Rec(
            num_items=num_items_with_special_tokens,
            hidden_dim=config["hidden_dim"],
            max_len=MAX_LEN,
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            dropout=BERT_DROPOUT,
            pad_token_id=pad_token_id
        ).to(device)

        # Train the model
        best_model = train_model_bert4rec(
            model, train_loader, val_loader, user_sequences_full_sets, num_items, num_items_with_special_tokens,
            pad_token_id, device, config, epochs=EPOCHS, patience=PATIENCE, lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY, top_k=TOP_K
        )

        # Evaluate the best model on the test set
        print(f"\nEvaluating {config['name']} on the test set...")
        test_recall, test_ndcg = evaluate_model_bert4rec(
            best_model, test_loader, user_sequences_full_sets, num_items, device, TOP_K, num_negatives=100
        )

        print(f"\n--- Test Results for {config['name']} ---")
        print(f"Test Recall@{TOP_K}: {test_recall:.4f}")
        print(f"Test NDCG@{TOP_K}: {test_ndcg:.4f}")

        # Store results
        results.append({
            "Model": config["name"],
            "Hidden Dim": config["hidden_dim"],
            "Layers": config["n_layers"],
            "Heads": config["n_heads"],
            "Mask Prob": config["mask_prob"],
            f"Recall@{TOP_K}": round(test_recall, 4),
            f"NDCG@{TOP_K}": round(test_ndcg, 4)
        })

    # Print and save overall comparison results
    print(f"\n--- Overall Comparison ---")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=f"NDCG@{TOP_K}", ascending=False)
    print(results_df.to_string(index=False))

    # Generate and save comparison plot with table
    try:
        fig = plt.figure(figsize=(12, 8))
        ax_bar = fig.add_axes([0.1, 0.4, 0.8, 0.5])
        x = np.arange(len(results_df))
        width = 0.35
        recall_col = f"Recall@{TOP_K}"
        ndcg_col = f"NDCG@{TOP_K}"
        rects1 = ax_bar.bar(x - width/2, results_df[recall_col], width, label=f'Recall@{TOP_K}', color='tab:blue')
        rects2 = ax_bar.bar(x + width/2, results_df[ndcg_col], width, label=f'NDCG@{TOP_K}', color='tab:orange')

        ax_bar.set_ylabel('Scores')
        ax_bar.set_title(f'BERT4Rec (ML-1M, MAX_LEN={MAX_LEN})')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(results_df['Model'])
        ax_bar.legend()
        ax_bar.bar_label(rects1, padding=3, fmt='%.4f', fontsize=8)
        ax_bar.bar_label(rects2, padding=3, fmt='%.4f', fontsize=8)
        max_score = max(results_df[recall_col].max(), results_df[ndcg_col].max()) if not results_df.empty else 0.1
        ax_bar.set_ylim(0, min(max(0.1, max_score * 1.2), 1.0))
        ax_bar.grid(True, axis='y', linestyle='--', alpha=0.7)

        ax_table = fig.add_axes([0.1, 0.05, 0.8, 0.25])
        ax_table.axis('off')
        table_data = results_df[['Model', 'Hidden Dim', 'Layers', 'Heads', 'Mask Prob', recall_col, ndcg_col]].values
        col_labels = ['Model', 'Hidden Dim', 'Layers', 'Heads', 'Mask Prob', f'Recall@{TOP_K}', f'NDCG@{TOP_K}']
        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)

        plot_filename = f"bert4rec_comparison_plot.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Comparison plot with table saved to {plot_filename}")
        plt.close()

        print("\n--- Comparison Table (Markdown) ---")
        print(results_df.to_markdown(index=False))

    except Exception as e:
        print(f"\nWarning: Could not generate or save plot. Error: {e}")


# Step 8: Run
if __name__ == "__main__":
    if not os.path.exists("ratings.dat"):
        print("Error: 'ratings.dat' not found in the current directory.")
        print("Please download the MovieLens 1M dataset and place 'ratings.dat' here.")
    else:
        main()
