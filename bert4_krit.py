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
from torch.cuda.amp import autocast, GradScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration class to manage hyperparameters
class Config:
    """Centralized configuration for BERT4Rec hyperparameters."""
    def __init__(self):
        # Data preprocessing
        self.max_len = 200
        self.min_interactions = 5
        self.rating_threshold = 4
        self.train_split = 0.7
        self.val_split = 0.15
        self.test_split = 0.15

        # Dataset
        self.train_batch_size = 128
        self.eval_batch_size = 512
        self.mask_prob = 0.3
        self.aug_prob = 0.1
        self.num_negatives = 20
        self.num_negatives_eval = 99

        # Model
        self.dropout = 0.2
        self.top_k = 10

        # Training
        self.epochs = 200
        self.patience = 5
        self.grad_clip = 5.0
        self.weight_decay = 0.0
        self.ranking_margin = 1.0
        self.warmup_ratio = 0.1
        self.scheduler_factor = 0.5
        self.scheduler_patience = 1

        # Model configurations with associated learning rates
        self.model_configs = [
            {"name": "BERT_Small", "hidden_dim": 64, "n_layers": 2, "n_heads": 2, "lr": 0.0012},
            {"name": "BERT_Medium", "hidden_dim": 128, "n_layers": 2, "n_heads": 4, "lr": 9e-4},
            {"name": "BERT_Large", "hidden_dim": 256, "n_layers": 3, "n_heads": 8, "lr": 7e-4},
        ]
        self._validate_model_configs()

    def _validate_model_configs(self):
        """Validate that hidden_dim is divisible by n_heads for each model config."""
        for config in self.model_configs:
            hidden_dim = config["hidden_dim"]
            n_heads = config["n_heads"]
            if hidden_dim % n_heads != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads}) "
                    f"for model config {config['name']}"
                )

# Data preprocessing class
class DataProcessor:
    """Handles loading and preprocessing of the MovieLens 1M dataset."""
    def __init__(self, config):
        self.config = config
        self.dataset_path = "ratings.dat"
        self.ratings = None
        self.user_sequences_full = None

    def load_data(self):
        """Load the MovieLens 1M dataset."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                "\nratings.dat not found in the current directory.\n"
                "Please download the MovieLens 1M dataset (ml-1m.zip) from:\n"
                "https://grouplens.org/datasets/movielens/1m/\n"
                "Extract it and place 'ratings.dat' in the same directory as this script."
            )
        print("Loading MovieLens 1M dataset...")
        self.ratings = pd.read_csv(
            self.dataset_path,
            sep="::",
            engine='python',
            names=['userId', 'movieId', 'rating', 'timestamp']
        )
        print(f"Original interactions: {len(self.ratings)}")

    def filter_data(self):
        """Filter ratings >= 4 and users/items with < 5 interactions."""
        # Filter ratings
        self.ratings = self.ratings[self.ratings['rating'] >= self.config.rating_threshold].copy()
        print(f"Interactions after filtering (rating >= {self.config.rating_threshold}): {len(self.ratings)}")

        # Filter users and items with < 5 interactions
        print("Filtering out users/items with few interactions...")
        user_counts = self.ratings['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.config.min_interactions].index
        self.ratings = self.ratings[self.ratings['userId'].isin(valid_users)]
        print(f"Interactions after user filtering (<{self.config.min_interactions} interactions): {len(self.ratings)}")

        item_counts = self.ratings['movieId'].value_counts()
        valid_items = item_counts[item_counts >= self.config.min_interactions].index
        self.ratings = self.ratings[self.ratings['movieId'].isin(valid_items)]
        print(f"Interactions after item filtering (<{self.config.min_interactions} interactions): {len(self.ratings)}")

        # Re-check user counts
        user_counts = self.ratings['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.config.min_interactions].index
        self.ratings = self.ratings[self.ratings['userId'].isin(valid_users)]
        print(f"Interactions after final user filtering: {len(self.ratings)}")

    def map_ids(self):
        """Map user and item IDs to contiguous indices."""
        print("Mapping user and item IDs...")
        unique_users = sorted(self.ratings['userId'].unique())
        unique_items = sorted(self.ratings['movieId'].unique())

        self.user_map = {user_id: i for i, user_id in enumerate(unique_users)}
        self.item_map = {item_id: i + 1 for i, item_id in enumerate(unique_items)}
        self.item_rev_map = {v: k for k, v in self.item_map.items()}

        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        self.pad_token_id = 0
        self.mask_token_id = self.num_items + 1
        self.num_items_with_special_tokens = self.num_items + 2

        print(f"Num users after filtering: {self.num_users}, Num items after filtering: {self.num_items}")
        print(f"Num items including special tokens: {self.num_items_with_special_tokens}")
        print(f"PAD ID: {self.pad_token_id}, MASK ID: {self.mask_token_id}")

        self.ratings['userId_mapped'] = self.ratings['userId'].map(self.user_map)
        self.ratings['movieId_mapped'] = self.ratings['movieId'].map(self.item_map)

    def create_sequences_full(self):
        """Create chronologically ordered sequences per user."""
        print("Sorting interactions by user and timestamp...")
        self.ratings = self.ratings.sort_values(['userId_mapped', 'timestamp'])

        print("Grouping interactions into sequences...")
        self.user_sequences_full = self.ratings.groupby('userId_mapped')['movieId_mapped'].apply(list).to_dict()

    def analyze_sequences(self):
        """Analyze sequence length distribution and set max_len."""
        print("Analyzing sequence length distribution...")
        sequence_lengths = [len(seq) for seq in self.user_sequences_full.values()]
        print(f"Sequence length statistics:")
        print(f"  Min: {min(sequence_lengths)}")
        print(f"  Max: {max(sequence_lengths)}")
        print(f"  Median: {np.median(sequence_lengths)}")
        print(f"  Mean: {np.mean(sequence_lengths):.2f}")
        print(f"Setting MAX_LEN={self.config.max_len} as per BERT4Rec paper for ML-1M")

    def split_data(self):
        """Split users into train/val/test sets."""
        all_user_ids = list(self.user_sequences_full.keys())
        train_user_ids, temp_user_ids = train_test_split(
            all_user_ids,
            test_size=(self.config.val_split + self.config.test_split),
            random_state=42
        )
        val_user_ids, test_user_ids = train_test_split(
            temp_user_ids,
            test_size=self.config.test_split / (self.config.val_split + self.config.test_split),
            random_state=42
        )
        print(f"Splitting users: {len(train_user_ids)} train, {len(val_user_ids)} val, {len(test_user_ids)} test")
        return train_user_ids, val_user_ids, test_user_ids

    def create_sequences(self, user_ids):
        """Create overlapping sequences for users using a sliding window."""
        user_sequences = []
        for user_id in tqdm(user_ids, desc="Creating sequences"):
            seq = self.user_sequences_full[user_id]
            if len(seq) < 2:
                continue
            step_size = self.config.max_len // 2
            for start in range(0, len(seq), step_size):
                end = min(start + self.config.max_len, len(seq))
                if end - start >= 2:
                    user_sequences.append((user_id, seq[start:end]))
        return user_sequences

    def preprocess(self):
        """Run the full preprocessing pipeline."""
        self.load_data()
        self.filter_data()
        self.map_ids()
        self.create_sequences_full()
        self.analyze_sequences()
        train_user_ids, val_user_ids, test_user_ids = self.split_data()
        train_sequences = self.create_sequences(train_user_ids)
        val_sequences = self.create_sequences(val_user_ids)
        test_sequences = self.create_sequences(test_user_ids)
        print(f"Created sequences: {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test")
        return (train_sequences, val_sequences, test_sequences,
                self.user_map, self.item_map, self.item_rev_map,
                self.num_users, self.num_items_with_special_tokens,
                self.pad_token_id, self.mask_token_id)

# Dataset class for BERT4Rec
class BERT4RecDataset(Dataset):
    """Dataset class for BERT4Rec, handling sequence augmentation, masking, and negative sampling."""
    def __init__(self, user_sequences, config, mask_token_id, pad_token_id, num_items, mode='train'):
        self.sequences = user_sequences
        self.max_len = config.max_len
        self.mask_prob = config.mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.num_items = num_items
        self.mode = mode
        self.aug_prob = config.aug_prob
        self.num_negatives = config.num_negatives

    def __len__(self):
        return len(self.sequences)

    def _augment_sequence(self, seq):
        """Apply data augmentation (shuffle or drop) to the sequence."""
        seq = seq.copy()
        seq_len = len(seq)
        if self.mode != 'train' or random.random() >= self.aug_prob:
            return seq

        if random.random() < 0.5 and seq_len > 2:
            seq = seq.copy()
            random.shuffle(seq[:-1])
        elif seq_len > 2:
            drop_idx = random.randrange(1, seq_len - 1)
            seq.pop(drop_idx)
        return seq

    def _mask_sequence(self, seq, input_ids, labels):
        """Mask items in the sequence for training."""
        masked_indices = []
        for i in range(len(seq)):
            if random.random() < self.mask_prob:
                masked_indices.append(i)
                labels[i] = input_ids[i]
                mask_decision = random.random()
                if mask_decision < 0.8:
                    input_ids[i] = self.mask_token_id
                elif mask_decision < 0.9:
                    random_item_id = random.randint(1, self.num_items)
                    input_ids[i] = random_item_id

        if len(seq) > 0 and not masked_indices:
            mask_pos = random.randrange(len(seq))
            labels[mask_pos] = input_ids[mask_pos]
            input_ids[mask_pos] = self.mask_token_id

    def _generate_negative_samples(self, seq):
        """Generate negative samples for ranking loss."""
        seq_set = set(seq)
        negative_samples = []
        while len(negative_samples) < self.num_negatives:
            neg_item = random.randint(1, self.num_items)
            if neg_item not in seq_set and neg_item != self.pad_token_id and neg_item != self.mask_token_id:
                negative_samples.append(neg_item)
        return negative_samples

    def __getitem__(self, idx):
        _, seq = self.sequences[idx]
        seq = self._augment_sequence(seq)
        seq_len = len(seq)

        # Pad the sequence
        padding_len = self.max_len - len(seq)
        padded_seq = seq + [self.pad_token_id] * padding_len
        input_ids = np.array(padded_seq)
        labels = np.full(self.max_len, self.pad_token_id)

        if self.mode == 'train':
            self._mask_sequence(seq, input_ids, labels)
            negative_samples = self._generate_negative_samples(seq)
            return (torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long),
                    torch.tensor(negative_samples, dtype=torch.long))
        elif self.mode in ['val', 'test']:
            if len(seq) > 0:
                target_item = input_ids[len(seq) - 1]
                input_ids[len(seq) - 1] = self.mask_token_id
                labels[len(seq) - 1] = target_item
                return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_item, dtype=torch.long)
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(self.pad_token_id, dtype=torch.long)

# BERT4Rec model classes
class PositionalEmbedding(nn.Module):
    """Positional embedding layer for BERT4Rec."""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.pe.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        batch_size, seq_len = x.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.pe(position_ids)

class BERT4Rec(nn.Module):
    """BERT4Rec model for sequential recommendation."""
    def __init__(self, num_items, hidden_dim, max_len, n_layers, n_heads, dropout, pad_token_id):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.mask_token_id = -1

        print(f"Initializing BERT4Rec with hidden_dim={hidden_dim}, n_heads={n_heads}")  # Debug statement

        # Validate that hidden_dim is divisible by n_heads
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads}) "
                f"to ensure proper multi-head attention splitting."
            )

        # Embeddings
        self.item_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=pad_token_id)
        self.positional_embedding = PositionalEmbedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_layernorm = nn.LayerNorm(hidden_dim, eps=1e-12)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_dim, eps=1e-12)
        )

        # Output layers
        self.transformer_layernorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_items)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if module.weight.requires_grad:
                    module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                if module.bias.requires_grad:
                    module.bias.data.zero_()

    def forward(self, input_ids):
        attention_mask = (input_ids == self.pad_token_id)
        item_emb = self.item_embedding(input_ids)
        pos_emb = self.positional_embedding(input_ids)
        embedding = item_emb + pos_emb
        embedding = self.emb_layernorm(embedding)
        embedding = self.emb_dropout(embedding)

        transformer_output = self.transformer_encoder(
            embedding,
            src_key_padding_mask=attention_mask
        )
        transformer_output = self.transformer_layernorm(transformer_output)
        transformer_output = self.output_dropout(transformer_output)
        logits = self.output_layer(transformer_output)
        return logits

# Trainer class for training and evaluation
class Trainer:
    """Handles training, validation, and evaluation of the BERT4Rec model."""
    def __init__(self, model, train_loader, val_loader, test_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.pad_token_id = model.pad_token_id
        self.mask_token_id = model.mask_token_id
        self.num_items_all = model.output_layer.out_features

        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.model_configs[0]["lr"],  # Will be set per config
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True
        )
        self.scaler = GradScaler()
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, label_smoothing=0.1)

    def compute_ranking_loss(self, logits, target_item, negative_samples, pad_mask):
        """Compute BPR-style pairwise ranking loss."""
        batch_size = logits.size(0)
        loss = 0.0
        for i in range(batch_size):
            seq_len = (pad_mask[i] == 0).sum().item()
            if seq_len == 0:
                continue
            mask_pos = seq_len - 1
            scores = logits[i, mask_pos, :]
            true_score = scores[target_item[i]]
            neg_scores = scores[negative_samples[i]]
            loss += -torch.log(torch.sigmoid(true_score - neg_scores + self.config.ranking_margin).clamp(min=1e-7)).mean()
        return loss / batch_size if batch_size > 0 else 0

    def train_epoch(self, epoch, total_steps, warmup_steps, base_lr):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        step = epoch * len(self.train_loader)

        train_progress = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Training {self.model_name}")
        for batch_idx, (input_ids, labels, negative_samples) in enumerate(train_progress):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            negative_samples = negative_samples.to(self.device)
            pad_mask = (input_ids == self.pad_token_id)

            # Adjust learning rate for warmup
            global_step = step + batch_idx
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = base_lr * lr_scale
            else:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = base_lr

            self.optimizer.zero_grad()
            with autocast():
                logits = self.model(input_ids)
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                ce_loss = self.criterion_ce(logits_flat, labels_flat)
                rank_loss = self.compute_ranking_loss(logits, labels[:, -1], negative_samples, pad_mask)
                loss = ce_loss + 0.5 * rank_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            train_progress.set_postfix({'loss': loss.item(), 'lr': self.optimizer.param_groups[0]['lr']})

        return total_loss / len(self.train_loader)

    def evaluate(self, data_loader, mode='val'):
        """Evaluate the model on validation or test data."""
        self.model.eval()
        recall_list = []
        ndcg_list = []

        eval_progress = tqdm(data_loader, total=len(data_loader), desc=f"Evaluating ({mode})")
        with torch.no_grad():
            with autocast():
                for input_ids, target_item in eval_progress:
                    input_ids, target_item = input_ids.to(self.device), target_item.to(self.device)
                    logits = self.model(input_ids)
                    batch_recalls = []
                    batch_ndcgs = []

                    for i in range(input_ids.size(0)):
                        seq_len_tensor = (input_ids[i] != self.pad_token_id).sum()
                        seq_len = seq_len_tensor.item()
                        if seq_len == 0:
                            continue

                        mask_pos = seq_len - 1
                        scores = logits[i, mask_pos, :].squeeze()
                        true_item_id = target_item[i].item()
                        if true_item_id == self.pad_token_id:
                            continue

                        # Negative sampling
                        seq_set = set(input_ids[i].cpu().numpy())
                        negative_samples = []
                        while len(negative_samples) < self.config.num_negatives_eval:
                            neg_item = random.randint(1, self.num_items_all - 2)
                            if neg_item not in seq_set and neg_item != self.pad_token_id and neg_item != self.mask_token_id:
                                negative_samples.append(neg_item)

                        # Rank candidates
                        candidate_items = [true_item_id] + negative_samples
                        candidate_scores = scores[candidate_items]
                        _, indices = torch.topk(candidate_scores, k=self.config.top_k, dim=-1)
                        ranked_items = [candidate_items[idx] for idx in indices.cpu().numpy()]

                        # Compute metrics
                        recall = 1.0 if true_item_id in ranked_items else 0.0
                        batch_recalls.append(recall)

                        ndcg = 0.0
                        if recall > 0:
                            rank = ranked_items.index(true_item_id)
                            ndcg = 1.0 / math.log2(rank + 2)
                        batch_ndcgs.append(ndcg)

                    if batch_recalls:
                        recall_list.extend(batch_recalls)
                    if batch_ndcgs:
                        ndcg_list.extend(batch_ndcgs)

        avg_recall = np.mean(recall_list) if recall_list else 0.0
        avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
        return avg_recall, avg_ndcg

    def train(self, config):
        """Train the model with the given configuration."""
        self.model_name = config['name']
        base_lr = config['lr']
        print(f"\n=== Training {self.model_name} ===")
        print(f"Config: {config}")
        print(f"Starting training (LR={base_lr}, patience={self.config.patience} on NDCG@{self.config.top_k})...\n")

        # Reset optimizer with the correct learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr

        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = int(self.config.warmup_ratio * total_steps)

        best_val_ndcg = -1
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            avg_loss = self.train_epoch(epoch, total_steps, warmup_steps, base_lr)
            print(f"Training Loss: {avg_loss:.4f}")

            val_recall, val_ndcg = self.evaluate(self.val_loader, mode='val')
            print(f"Val Recall@{self.config.top_k}: {val_recall:.4f}")
            print(f"Val NDCG@{self.config.top_k}: {val_ndcg:.4f}")

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_model_state = self.model.state_dict()
                epochs_no_improve = 0
                print(f"New best model saved based on validation NDCG@{self.config.top_k}!\n")
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation NDCG@{self.config.top_k}, "
                      f"epochs without improvement: {epochs_no_improve}/{self.config.patience}\n")
                if epochs_no_improve >= self.config.patience:
                    print("Early stopping triggered")
                    break

            self.scheduler.step(val_ndcg)

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"Training completed for {self.model_name}, best model state loaded based on validation NDCG@{self.config.top_k}.")
        else:
            print(f"Training completed for {self.model_name} without improvement or early stopping hit max epochs.")

        test_recall, test_ndcg = self.evaluate(self.test_loader, mode='test')
        print(f"\n--- Test Results for {self.model_name} ---")
        print(f"Test Recall@{self.config.top_k}: {test_recall:.4f}")
        print(f"Test NDCG@{self.config.top_k}: {test_ndcg:.4f}")

        return test_recall, test_ndcg

# Visualization class for results
class Visualizer:
    """Handles visualization and saving of comparison results."""
    @staticmethod
    def plot_comparison(results, config):
        """Generate a comparison table and bar plot for the results."""
        df = pd.DataFrame(results)
        print("\n--- Overall Comparison ---")
        print(df.to_string(index=False))
        df.to_csv("bert4rec_comparison.csv", index=False)
        print("\nComparison results saved to bert4rec_comparison.csv")

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            ax1.axis("off")
            table = ax1.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.1, 1.1)
            ax1.set_title("BERT4Rec Configuration Comparison", fontsize=14, pad=20)

            x = np.arange(len(results))
            width = 0.35
            recall_col = f"Test Recall@{config.top_k}"
            ndcg_col = f"Test NDCG@{config.top_k}"
            rects1 = ax2.bar(x - width/2, df[recall_col], width, label=f"Recall@{config.top_k}", color="#6BAED6")
            rects2 = ax2.bar(x + width/2, df[ndcg_col], width, label=f"NDCG@{config.top_k}", color="#FDBB84")

            ax2.set_ylabel("Score", fontsize=12)
            ax2.set_title("Test Set Performance", fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels(df["Model"])
            ax2.legend(loc="upper right")
            ax2.grid(True, axis="y", linestyle="--", alpha=0.7)

            max_score = max(df[recall_col].max(), df[ndcg_col].max())
            ax2.set_ylim(0, min(max(0.1, max_score * 1.2), 1.0))

            ax2.bar_label(rects1, padding=3, fmt='%.4f', fontsize=8)
            ax2.bar_label(rects2, padding=3, fmt='%.4f', fontsize=8)

            plt.tight_layout(pad=2.0)
            plt.savefig("bert4rec_comparison_plot.png", dpi=300)
            plt.close()
            print("Comparison plot saved as bert4rec_comparison_plot.png")
        except Exception as e:
            print(f"\nWarning: Could not generate plot. Error: {e}")
            print("Ensure matplotlib is installed and working correctly.")

# Main function to orchestrate the workflow
def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preprocess data
    processor = DataProcessor(config)
    try:
        (train_sequences, val_sequences, test_sequences,
         user_map, item_map, item_rev_map,
         num_users, num_items_with_special_tokens,
         pad_token_id, mask_token_id) = processor.preprocess()
    except FileNotFoundError as e:
        print(e)
        return

    print(f"\nNumber of items (incl. special tokens): {num_items_with_special_tokens}")
    print(f"Number of users (after filtering): {num_users}")

    num_items = len(item_map)
    num_workers = 2 if device.type == 'cuda' else 0

    # Create datasets and data loaders
    train_dataset = BERT4RecDataset(train_sequences, config, mask_token_id, pad_token_id, num_items, mode='train')
    val_dataset = BERT4RecDataset(val_sequences, config, mask_token_id, pad_token_id, num_items, mode='val')
    test_dataset = BERT4RecDataset(test_sequences, config, mask_token_id, pad_token_id, num_items, mode='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Train and evaluate models
    results = []
    for model_config in config.model_configs:
        print(f"\n--- Starting Configuration: {model_config['name']} ---")
        model = BERT4Rec(
            num_items=num_items_with_special_tokens,
            hidden_dim=model_config["hidden_dim"],
            max_len=config.max_len,
            n_layers=model_config["n_layers"],
            n_heads=model_config["n_heads"],
            dropout=config.dropout,
            pad_token_id=pad_token_id
        ).to(device)
        model.mask_token_id = mask_token_id
        model.pad_token_id = pad_token_id

        trainer = Trainer(model, train_loader, val_loader, test_loader, config, device)
        test_recall, test_ndcg = trainer.train(model_config)

        results.append({
            "Model": model_config["name"],
            "Hidden Dim": model_config["hidden_dim"],
            "Layers": model_config["n_layers"],
            "Heads": model_config["n_heads"],
            f"Test Recall@{config.top_k}": round(test_recall, 4),
            f"Test NDCG@{config.top_k}": round(test_ndcg, 4)
        })

    # Visualize results
    Visualizer.plot_comparison(results, config)

if __name__ == "__main__":
    main()