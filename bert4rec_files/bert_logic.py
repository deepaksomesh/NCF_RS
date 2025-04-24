import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


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