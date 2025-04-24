import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import random
import config



# Step 6: Define evaluation function for BERT4Rec with Negative Sampling
def evaluate_model_bert4rec(model, data_loader, user_sequences_full_sets, num_items, device, top_k=config.TOP_K, num_negatives=100):
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