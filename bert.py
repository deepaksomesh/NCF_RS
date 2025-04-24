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
import os # For checking file existence

# Step 1: Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Constants ---
MAX_LEN = 200         # Maximum sequence length for BERT4Rec (Requirement: fixed length, e.g., 20 - adjust if needed)
# The paper mentions using N=200 for ML-1M, N=50 for Beauty/Steam.
# Let's use 50 as a reasonable default, adjustable if needed.
MASK_PROB = 0.2     # Probability of masking an item in the sequence for training (Paper explored 0.1-0.9, found 0.2 optimal for ML-1M, 0.4 for Steam, 0.6 for Beauty)
# We'll use 0.2 as a default, can be made configurable if needed per dataset.
# Model Hyperparameters (These will be varied in configurations)
# BERT_HIDDEN_DIM = 128 # Defined in configs
# BERT_ATTENTION_HEADS = 4 # Defined in configs
# BERT_LAYERS = 2 # Defined in configs
BERT_DROPOUT = 0.2 # Kept constant for simplicity, but could be varied (Paper likely used 0.1 or 0.2 based on BERT practices)
# Training Hyperparameters
TRAIN_BATCH_SIZE = 256 # Paper used 256, smaller batch size might be needed depending on GPU memory
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 0.001 # Paper used 1e-4 (0.0001), this might be too high. Let's adjust. LR = 1e-4
LEARNING_RATE = 1e-4
EPOCHS = 200 # Max epochs, early stopping will likely trigger sooner (Paper doesn't specify epochs, but likely ran until convergence/early stopping)
PATIENCE = 10 # Patience for early stopping based on Val NDCG@10 (Increased patience slightly)
TOP_K = 10 # For Recall@K and NDCG@K
# SCHEDULER_STEP_SIZE = 3 # Paper mentions linear decay, not step decay. Let's remove StepLR and implement linear decay or use AdamW.
# SCHEDULER_GAMMA = 0.1
# AdamW is often preferred for Transformer models
WEIGHT_DECAY = 0.01 # Paper mentions l2 weight decay of 0.01


# Step 2: Define function to load and preprocess data for BERT4Rec
def load_and_preprocess_data_bert4rec(max_len=MAX_LEN):
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


# Step 3: Define custom Dataset class for BERT4Rec
class BERT4RecDataset(Dataset):
    """
    A custom PyTorch Dataset class for BERT4Rec, conforming to requirements.
    Handles padding and masking strategy based on mode (train/val/test).
    Yields user_id in val/test modes for negative sampling.
    """
    def __init__(self, user_sequences, max_len, mask_prob, mask_token_id, pad_token_id, num_items, mode='train'):
        self.user_ids = list(user_sequences.keys()) # These are mapped user IDs
        self.sequences_data = [user_sequences[uid] for uid in self.user_ids]
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.num_items = num_items # Number of actual items (1 to num_items)
        self.mode = mode
        # Create a set of all valid item IDs for faster negative sampling checks if needed here
        # self.all_items = set(range(1, num_items + 1)) # Not strictly needed here, but maybe in eval

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx] # Get the mapped user ID

        if self.mode == 'train':
            seq = self.sequences_data[idx]
        else: # val or test
            seq, target_item = self.sequences_data[idx]

        seq_len = len(seq)
        padding_len = max(0, self.max_len - seq_len)
        padded_seq = seq + [self.pad_token_id] * padding_len
        input_ids = np.array(padded_seq[:self.max_len]) # Ensure length is exactly max_len
        labels = np.full(self.max_len, self.pad_token_id)

        if self.mode == 'train':
            output_seq = input_ids.copy()
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
                mask_pos = seq_len - 1
                labels[mask_pos] = input_ids[mask_pos]
                output_seq[mask_pos] = self.mask_token_id

            return torch.tensor(output_seq, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

        elif self.mode in ['val', 'test']:
            output_seq = input_ids.copy()
            if seq_len > 0:
                mask_pos = seq_len - 1
                output_seq[mask_pos] = self.mask_token_id
                # Return user_id along with sequence and target
                return torch.tensor(output_seq, dtype=torch.long), torch.tensor(target_item, dtype=torch.long), torch.tensor(user_id, dtype=torch.long)
            else:
                 # Return dummy user_id -1 if sequence is empty
                 return torch.tensor(output_seq, dtype=torch.long), torch.tensor(self.pad_token_id, dtype=torch.long), torch.tensor(-1, dtype=torch.long)

# Step 4: Define the BERT4Rec model
class PositionalEmbedding(nn.Module):
    """Learnable Positional Embedding"""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        # Create position indices [0, 1, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.pe(position_ids)

class BERT4Rec(nn.Module):
    """
    BERT4Rec model implementation using Transformer Encoder. Conforms to requirements.

    Args:
        num_items (int): Total number of items including special tokens (PAD=0, MASK=N+1).
        hidden_dim (int): Dimension of the transformer embeddings and hidden layers.
        max_len (int): Maximum sequence length.
        n_layers (int): Number of transformer encoder layers.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        pad_token_id (int): ID for the padding token.
    """
    def __init__(self, num_items, hidden_dim, max_len, n_layers, n_heads, dropout, pad_token_id):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.mask_token_id = num_items - 1 # MASK ID is the last index (num_items includes PAD and MASK)

        # --- REQUIREMENT: Item and Positional Embeddings ---
        # Item embedding layer - padding_idx tells the layer to ignore gradients for PAD token
        # and initialize its embedding to zeros (though we might override with init)
        self.item_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=pad_token_id)
        # Positional embedding layer (learnable)
        self.positional_embedding = PositionalEmbedding(max_len, hidden_dim)

        # --- REQUIREMENT: Dropout and Layer Normalization ---
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_layernorm = nn.LayerNorm(hidden_dim, eps=1e-12) # LayerNorm after embedding sum

        # --- REQUIREMENT: Transformer Encoder ---
        # Define a single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,         # Input feature dimension
            nhead=n_heads,              # Number of attention heads
            dim_feedforward=hidden_dim * 4, # Standard BERT FFN size
            dropout=dropout,            # Dropout within the layer
            activation='gelu',          # GELU activation (as in BERT)
            batch_first=True,           # Input/output format is (batch, seq, feature)
            norm_first=False            # Post-LN: Apply LN after attention/FFN (like original Transformer)
        )
        # Stack multiple encoder layers
        # Add final LayerNorm after the stack (common practice)
        encoder_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=encoder_norm            # Final normalization layer
            )

        # --- REQUIREMENT: Final softmax layer over item vocabulary ---
        # Output layer projects transformer output to item score dimension
        self.output_layer = nn.Linear(hidden_dim, num_items) # Predict scores for all items (including PAD, MASK)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize weights similar to BERT - Normal(0.0, 0.02)
        # Based on paper Section 4.3: truncated normal distribution [-0.02, 0.02]
        stddev = 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if module.weight.requires_grad:
                     # Truncated normal initialization
                     torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=stddev, a=-2*stddev, b=2*stddev)
            elif isinstance(module, nn.LayerNorm):
                # Initialize LN bias to 0, weight to 1
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            # Initialize linear biases to zero
            if isinstance(module, nn.Linear) and module.bias is not None:
                 if module.bias.requires_grad:
                    module.bias.data.zero_()
        # Special handling for positional embedding? Paper doesn't specify, common practice is normal or xavier.
        # Let's use the same truncated normal for consistency.
        if hasattr(self, 'positional_embedding') and isinstance(self.positional_embedding.pe, nn.Embedding):
             torch.nn.init.trunc_normal_(self.positional_embedding.pe.weight, mean=0.0, std=stddev, a=-2*stddev, b=2*stddev)
        # Ensure pad embedding is zero after initialization
        if hasattr(self.item_embedding, 'padding_idx') and self.item_embedding.padding_idx is not None:
             with torch.no_grad():
                self.item_embedding.weight[self.item_embedding.padding_idx].fill_(0)


    def forward(self, input_ids):
        # input_ids shape: (batch_size, max_len)

        # 1. Create attention mask for padding tokens
        # TransformerEncoder expects `src_key_padding_mask`: (batch_size, seq_len)
        # `True` indicates positions that should be *ignored* (masked) by attention.
        attention_mask = (input_ids == self.pad_token_id) # True for PAD tokens

        # 2. Get embeddings
        item_emb = self.item_embedding(input_ids) # Shape: (batch_size, max_len, hidden_dim)
        pos_emb = self.positional_embedding(input_ids) # Shape: (batch_size, max_len, hidden_dim)

        # 3. Combine embeddings + LayerNorm + Dropout
        embedding = item_emb + pos_emb
        embedding = self.emb_layernorm(embedding)
        embedding = self.emb_dropout(embedding) # Apply dropout *after* LN and sum

        # 4. Pass through Transformer Encoder
        # Pass the input embedding and the padding mask
        transformer_output = self.transformer_encoder(
            src=embedding,
            src_key_padding_mask=attention_mask
        ) # Shape: (batch_size, max_len, hidden_dim)

        # 5. Project to item vocabulary
        logits = self.output_layer(transformer_output) # Shape: (batch_size, max_len, num_items)

        return logits

# # Step 5: Define training function for BERT4Rec
# def train_model_bert4rec(model, train_loader, val_loader, num_items_with_special_tokens, pad_token_id, device, config, epochs=EPOCHS, patience=PATIENCE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
#     """
#     Trains the BERT4Rec model using specified requirements.
#     - Masked Language Modeling objective (CrossEntropyLoss)
#     - AdamW optimizer with weight decay
#     - No scheduler (or could add linear decay later if needed)
#     - Early stopping based on validation NDCG@10
#     """
#     model_name = config['name']
#     print(f"\n=== Training {model_name} ===")
#     print(f"Config: {config}")
#     print(f"Starting training (LR={lr}, WeightDecay={weight_decay}, patience={patience} on NDCG@{TOP_K})...\n")
#
#     # --- REQUIREMENT: Masked language modeling objective ---
#     # Use CrossEntropyLoss, ignore padding tokens in the labels
#     criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
#
#     # --- REQUIREMENT: Adam optimizer (Using AdamW as it's often better for Transformers) ---
#     # Paper mentions Adam with l2 decay 0.01 and linear LR decay. AdamW incorporates weight decay correctly.
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#     # --- REQUIREMENT: Learning rate scheduling ---
#     # Paper mentions linear decay. We can implement this or just use AdamW with fixed LR for simplicity first.
#     # Let's omit the scheduler for now and rely on AdamW + early stopping.
#     # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA) # Removed
#
#     # --- REQUIREMENT: Implement early stopping based on validation NDCG@10 ---
#     best_val_ndcg = -1
#     epochs_no_improve = 0
#     best_model_state = None
#
#     for epoch in range(epochs):
#         print(f"Epoch {epoch+1}/{epochs}")
#         model.train()
#         total_loss = 0
#
#         train_progress = tqdm(train_loader, total=len(train_loader), desc=f"Training {model_name}")
#         for input_ids, labels in train_progress:
#             input_ids, labels = input_ids.to(device), labels.to(device)
#             # input_ids: (batch_size, max_len) - sequence with masks
#             # labels: (batch_size, max_len) - original item IDs at masked positions, pad_token_id elsewhere
#
#             optimizer.zero_grad()
#             logits = model(input_ids) # Shape: (batch_size, max_len, num_items)
#
#             # Reshape for CrossEntropyLoss:
#             # Logits need shape (N, C) where N is total number of tokens to predict, C = num_items
#             # Labels need shape (N)
#             # Loss is computed only where labels != pad_token_id
#             logits_flat = logits.view(-1, logits.size(-1)) # Shape: (batch_size * max_len, num_items)
#             labels_flat = labels.view(-1) # Shape: (batch_size * max_len)
#
#             loss = criterion(logits_flat, labels_flat)
#             loss.backward()
#             # Gradient clipping (mentioned in paper Sec 4.3, threshold 5)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()
#
#             total_loss += loss.item()
#             # train_progress.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]}) # Old scheduler
#             train_progress.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
#
#
#         avg_loss = total_loss / len(train_loader)
#         print(f"Training Loss: {avg_loss:.4f}")
#
#         # Validation phase - Evaluate Recall@K and NDCG@K
#         model.eval()
#         val_recall, val_ndcg = evaluate_model_bert4rec(model, val_loader, num_items_with_special_tokens, device, top_k=TOP_K)
#         print(f"Val Recall@{TOP_K}: {val_recall:.4f}")
#         print(f"Val NDCG@{TOP_K}: {val_ndcg:.4f}")
#
#         # Early stopping logic
#         current_metric = val_ndcg # Use NDCG@10 for early stopping
#         if current_metric > best_val_ndcg:
#             best_val_ndcg = current_metric
#             best_model_state = model.state_dict() # Save the best model state
#             epochs_no_improve = 0
#             print(f"New best model found! Validation NDCG@{TOP_K}: {best_val_ndcg:.4f}\n")
#             # Save checkpoint (optional)
#             # torch.save(best_model_state, f"{model_name}_best_checkpoint.pth")
#         else:
#             epochs_no_improve += 1
#             print(f"No improvement in validation NDCG@{TOP_K}. Epochs without improvement: {epochs_no_improve}/{patience}\n")
#             if epochs_no_improve >= patience:
#                 print("Early stopping triggered.")
#                 break
#
#         # Step the scheduler if using one
#         # scheduler.step() # Removed
#
#     # Load the best model weights found during training before returning
#     if best_model_state:
#         print(f"Loading best model state with validation NDCG@{TOP_K}: {best_val_ndcg:.4f}")
#         model.load_state_dict(best_model_state)
#     else:
#         print("Training finished without improving on initial validation score or hit max epochs.")
#
#     return model

# --- Add the modified train_model_bert4rec function ---
# Need to update train_model to accept and pass necessary args for evaluation
def train_model_bert4rec(model, train_loader, val_loader,
                         user_sequences_full_sets, num_items, # Added for eval
                         num_items_with_special_tokens, pad_token_id, device, config,
                         epochs=EPOCHS, patience=PATIENCE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                         top_k=TOP_K): # Added top_k
    """
    Trains the BERT4Rec model using specified requirements.
    Now passes necessary args for negative sampling evaluation during validation.
    """
    model_name = config['name']
    print(f"\n=== Training {model_name} ===")
    print(f"Config: {config}")
    print(f"Starting training (LR={lr}, WeightDecay={weight_decay}, patience={patience} on NDCG@{top_k})...\n") # Use top_k

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_ndcg = -1
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        train_progress = tqdm(train_loader, total=len(train_loader), desc=f"Training {model_name}")
        # Note: train_loader yields only (input_ids, labels)
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

        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")

        # --- Validation phase - Call evaluate with negative sampling ---
        model.eval()
        # Pass the necessary arguments for negative sampling evaluation
        val_recall, val_ndcg = evaluate_model_bert4rec(
            model=model,
            data_loader=val_loader,
            user_sequences_full_sets=user_sequences_full_sets,
            num_items=num_items,
            device=device,
            top_k=top_k, # Use the passed top_k
            num_negatives=100
        )
        print(f"Val Recall@{top_k}: {val_recall:.4f}") # Use top_k
        print(f"Val NDCG@{top_k}: {val_ndcg:.4f}")   # Use top_k

        current_metric = val_ndcg # Use NDCG for stopping
        if current_metric > best_val_ndcg:
            best_val_ndcg = current_metric
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            print(f"New best model found! Validation NDCG@{top_k}: {best_val_ndcg:.4f}\n") # Use top_k
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation NDCG@{top_k}. Epochs without improvement: {epochs_no_improve}/{patience}\n") # Use top_k
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state:
        print(f"Loading best model state with validation NDCG@{top_k}: {best_val_ndcg:.4f}") # Use top_k
        model.load_state_dict(best_model_state)
    else:
        print("Training finished without improving on initial validation score or hit max epochs.")

    return model # Return the best model


# Step 6: Define evaluation function for BERT4Rec with Negative Sampling
def evaluate_model_bert4rec(model, data_loader, user_sequences_full_sets, num_items, device, top_k=TOP_K, num_negatives=100):
    """
    Evaluates the BERT4Rec model using Recall@K and NDCG@K with negative sampling.

    Args:
        model (nn.Module): Trained BERT4Rec model.
        data_loader (DataLoader): DataLoader for evaluation data ('val' or 'test' mode).
                                  Must yield (input_seq, target_item, user_id).
        user_sequences_full_sets (dict): Mapped User ID -> set of all mapped item IDs interacted with by the user.
        num_items (int): Total number of actual items (1 to num_items) for sampling range.
        device (torch.device): Device to run the model on.
        top_k (int): K for Recall@K and NDCG@K.
        num_negatives (int): Number of negative items to sample per positive item.

    Returns:
        recall (float): Average Recall@K across all users.
        ndcg (float): Average NDCG@K across all users.
    """
    model.eval()
    recall_list = []
    ndcg_list = []
    pad_token_id = model.pad_token_id
    mask_token_id = model.mask_token_id
    all_item_ids = set(range(1, num_items + 1)) # Set of all valid item IDs (1 to num_items)

    eval_progress = tqdm(data_loader, total=len(data_loader), desc="Evaluating (Neg Sampling)")
    with torch.no_grad():
        # The dataloader now yields user_id as the third element
        for input_ids, target_item_ids, user_ids_batch in eval_progress:
            input_ids = input_ids.to(device)
            target_item_ids = target_item_ids.to(device) # Shape: (batch_size)
            user_ids_batch = user_ids_batch.to(device)   # Shape: (batch_size)

            logits = model(input_ids) # Shape: (batch_size, max_len, num_items_with_special_tokens)

            batch_recalls = []
            batch_ndcgs = []

            for i in range(input_ids.size(0)): # Iterate through batch
                user_id = user_ids_batch[i].item()
                if user_id == -1: continue # Skip dummy users from empty sequences

                seq_with_mask = input_ids[i]
                true_target_item = target_item_ids[i].item()

                # Find the mask position (as before)
                mask_positions = (seq_with_mask == mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_positions) == 0:
                    actual_len = (input_ids[i] != pad_token_id).sum().item()
                    if actual_len > 0: mask_pos = actual_len - 1
                    else: continue
                else:
                    mask_pos = mask_positions[0].item()

                # Get scores for *all* items at the masked position
                all_scores = logits[i, mask_pos, :] # Shape: (num_items_with_special_tokens)

                # --- Negative Sampling ---
                historical_items = user_sequences_full_sets.get(user_id, set())
                # Add target item to history temporarily in case it was also in historical interactions (shouldn't happen with leave-one-out)
                # historical_items_for_sampling = historical_items | {true_target_item} # Ensure target isn't sampled

                # Determine items available for negative sampling
                possible_negatives = list(all_item_ids - historical_items)

                # Sample negative items
                if len(possible_negatives) < num_negatives:
                    # If fewer possible negatives than required, sample with replacement or take all
                    # Taking all available is safer to avoid infinite loops if history is huge
                    negatives = random.choices(possible_negatives, k=num_negatives) if possible_negatives else []
                    # print(f"Warning: User {user_id} has only {len(possible_negatives)} possible negatives. Sampling {len(negatives)}.")
                else:
                    negatives = random.sample(possible_negatives, num_negatives)

                # Create the candidate list: 1 positive + sampled negatives
                candidate_items = [true_target_item] + negatives
                # Ensure candidates are valid item IDs (should be, as sampled from all_item_ids)
                candidate_items = [item for item in candidate_items if 1 <= item <= num_items] # Filter just in case

                if not candidate_items: continue # Skip if something went wrong

                # Get scores ONLY for the candidate items
                candidate_scores = all_scores[candidate_items] # Shape: (len(candidate_items))

                # --- Ranking & Metrics ---
                # Get top K indices relative to the candidate_scores tensor
                _, top_k_indices_relative = torch.topk(candidate_scores, k=min(top_k, len(candidate_items)))

                # Map these relative indices back to actual item IDs
                top_k_items = [candidate_items[idx.item()] for idx in top_k_indices_relative]

                # Calculate Recall@K
                recall = 1.0 if true_target_item in top_k_items else 0.0
                batch_recalls.append(recall)

                # Calculate NDCG@K
                ndcg = 0.0
                if recall > 0:
                    # Find rank within the candidate list (0-based)
                    try:
                        # Find the rank within the *sorted* candidate scores (equivalent to rank in top_k_items if k is large enough)
                        # A simpler way: find index in the top_k_items list
                        rank = top_k_items.index(true_target_item)
                        ndcg = 1.0 / math.log2(rank + 2)
                    except ValueError:
                        # Should not happen if recall is 1.0, but safety check
                        pass
                batch_ndcgs.append(ndcg)

            if batch_recalls: recall_list.extend(batch_recalls)
            if batch_ndcgs: ndcg_list.extend(batch_ndcgs)

    avg_recall = np.mean(recall_list) if recall_list else 0.0
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0

    return avg_recall, avg_ndcg

# Step 7: Define main function for BERT4Rec
def main():
    """
    Main function to load data, train multiple BERT4Rec configurations,
    evaluate them using negative sampling, and save comparison results.
    """
    # Step 7.1: Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 7.2: Load and preprocess data for BERT4Rec
    print("Loading and preprocessing data for BERT4Rec...")
    try:
        # Update unpacking to include user_sequences_full_sets
        (train_sequences, val_sequences, test_sequences,
         user_map, item_map, item_rev_map,
         num_users, num_items_with_special_tokens,
         pad_token_id, mask_token_id,
         user_sequences_full_sets) = load_and_preprocess_data_bert4rec(max_len=MAX_LEN) # Ensure MAX_LEN is appropriate (e.g., 200 for ML-1M)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        # ... (error message remains the same) ...
        return

    num_items = len(item_map) # Actual number of items (excluding special tokens)
    print(f"\nNumber of items (incl. special tokens PAD={pad_token_id}, MASK={mask_token_id}): {num_items_with_special_tokens}")
    print(f"Number of users (after filtering): {num_users}")
    print(f"Actual item range: 1 to {num_items}")

    # Step 7.3: Create DataLoaders (pass num_items to Dataset)
    train_dataset = BERT4RecDataset(train_sequences, MAX_LEN, MASK_PROB, mask_token_id, pad_token_id, num_items, mode='train')
    val_dataset = BERT4RecDataset(val_sequences, MAX_LEN, MASK_PROB, mask_token_id, pad_token_id, num_items, mode='val')
    test_dataset = BERT4RecDataset(test_sequences, MAX_LEN, MASK_PROB, mask_token_id, pad_token_id, num_items, mode='test')

    num_workers = 2 if os.name == 'posix' else 0
    print(f"Using {num_workers} workers for DataLoaders.")
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    # Ensure batch_first=True in DataLoader if not default, though it usually is
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    # --- Model configurations (remains the same) ---
    model_configs = [
        {"name": "BERT4Rec-S", "hidden_dim": 64,  "n_layers": 4, "n_heads": 4},
        {"name": "BERT4Rec-M", "hidden_dim": 128, "n_layers": 4, "n_heads": 4},
        {"name": "BERT4Rec-L", "hidden_dim": 256, "n_layers": 4, "n_heads": 4},
    ]

    results = []
    for config in model_configs:
        print(f"\n--- Starting Configuration: {config['name']} ---")
        model = BERT4Rec(
            num_items=num_items_with_special_tokens,
            hidden_dim=config["hidden_dim"],
            max_len=MAX_LEN,
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            dropout=BERT_DROPOUT,
            pad_token_id=pad_token_id
        ).to(device)

        # Train the model (train function call remains the same)
        # Pass necessary args to train_model_bert4rec
        best_model = train_model_bert4rec(
             model=model,
             train_loader=train_loader,
             val_loader=val_loader, # val_loader is now used inside train_model
             # Pass args needed by evaluate_model_bert4rec called *within* train_model
             num_items_with_special_tokens=num_items_with_special_tokens, # needed by criterion
             pad_token_id=pad_token_id, # needed by criterion/eval
             device=device,
             config=config,
             epochs=EPOCHS,
             patience=PATIENCE,
             lr=LEARNING_RATE,
             weight_decay=WEIGHT_DECAY,
             # Add args needed specifically for evaluation *inside* train_model
             user_sequences_full_sets=user_sequences_full_sets,
             num_items=num_items,
             top_k=TOP_K
         )


        # Evaluate on the test set using the best model
        print(f"\nEvaluating {config['name']} on the test set...")
        # Update the call to evaluate_model_bert4rec
        test_recall, test_ndcg = evaluate_model_bert4rec(
            model=best_model,
            data_loader=test_loader,
            user_sequences_full_sets=user_sequences_full_sets, # Pass history sets
            num_items=num_items,                           # Pass num actual items
            device=device,
            top_k=TOP_K,
            num_negatives=100                              # Specify num negatives
        )

        print(f"\n--- Test Results for {config['name']} ---")
        print(f"Test Recall@{TOP_K}: {test_recall:.4f}")
        print(f"Test NDCG@{TOP_K}: {test_ndcg:.4f}")

        results.append({
            "Model": config["name"],
            "Hidden Dim": config["hidden_dim"],
            "Layers": config["n_layers"],
            "Heads": config["n_heads"],
            f"Recall@{TOP_K}": round(test_recall, 4),
            f"NDCG@{TOP_K}": round(test_ndcg, 4)
        })

    # --- Comparison table and plot (remains the same) ---
    print("\n--- Overall Comparison ---")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=f"NDCG@{TOP_K}", ascending=False)
    print(results_df.to_string(index=False))
    results_df.to_csv("bert4rec_comparison_results.csv", index=False)
    print("\nComparison results saved to bert4rec_comparison_results.csv")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(results_df))
        width = 0.35
        recall_col = f"Recall@{TOP_K}"
        ndcg_col = f"NDCG@{TOP_K}"
        rects1 = ax.bar(x - width/2, results_df[recall_col], width, label=f'Recall@{TOP_K}', color='tab:blue')
        rects2 = ax.bar(x + width/2, results_df[ndcg_col], width, label=f'NDCG@{TOP_K}', color='tab:orange')
        ax.set_ylabel('Scores')
        ax.set_title(f'BERT4Rec Model Comparison (ML-1M, MAX_LEN={MAX_LEN})') # Add MAX_LEN to title
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'])
        ax.legend()
        ax.bar_label(rects1, padding=3, fmt='%.4f', fontsize=8)
        ax.bar_label(rects2, padding=3, fmt='%.4f', fontsize=8)
        max_score = max(results_df[recall_col].max(), results_df[ndcg_col].max()) if not results_df.empty else 0.1
        ax.set_ylim(0, min(max(0.1, max_score * 1.2), 1.0))
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()
        plot_filename = "bert4rec_comparison_plot.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"Comparison plot saved to {plot_filename}")
        plt.close()
        print("\n--- Comparison Table ---")
        print(results_df.to_markdown(index=False))
    except Exception as e:
        print(f"\nWarning: Could not generate or save plot. Error: {e}")


# Step 8: Run the main function (remains the same, just calls the modified main)
if __name__ == "__main__":
    if not os.path.exists("ratings.dat"):
         print("\nError: 'ratings.dat' not found...") # Shortened message
         # ... (rest of error message) ...
    else:
        # IMPORTANT: Set MAX_LEN appropriately for ML-1M based on paper/memory
        MAX_LEN = 200
        main()