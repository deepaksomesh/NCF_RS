import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from evaluate import evaluate_model_bert4rec
import config


# --- Add the modified train_model_bert4rec function ---
# Need to update train_model to accept and pass necessary args for evaluation
def train_model_bert4rec(model, train_loader, val_loader,
                         user_sequences_full_sets, num_items, # Added for eval
                         num_items_with_special_tokens, pad_token_id, device, config,
                         epochs=config.EPOCHS, patience=config.PATIENCE, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
                         top_k=config.TOP_K): # Added top_k
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