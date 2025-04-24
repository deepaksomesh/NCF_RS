import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os # For checking file existence
from preprocess import load_and_preprocess_data_bert4rec
from bert_logic import BERT4RecDataset, BERT4Rec
from train import train_model_bert4rec
from evaluate import evaluate_model_bert4rec
from config import MASK_PROB, MAX_LEN, TRAIN_BATCH_SIZE, TOP_K, EVAL_BATCH_SIZE, EPOCHS, BERT_DROPOUT, PATIENCE, LEARNING_RATE, WEIGHT_DECAY


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
        {"name": "BERT4Rec-S", "hidden_dim": 64, "n_layers": 4, "n_heads": 4},
        {"name": "BERT4Rec-M", "hidden_dim": 128, "n_layers": 4, "n_heads": 4},
        {"name": "BERT4Rec-L", "hidden_dim": 256, "n_layers": 4, "n_heads": 4}
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
        MAX_LEN = 200 # Set MAX_LEN = 200 as suggested by paper for ML-1M
        print(f"*** Running with MAX_LEN = {MAX_LEN} ***")
        main()