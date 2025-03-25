import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============
# 1. Define the NCF model class
# ============
# This class implements the Neural Collaborative Filtering model with GMF and MLP branches.
# It combines embeddings, matrix factorization, and a neural network for prediction.

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=32, mlp_layers=[64, 32, 16]):
        super(NCF, self).__init__()
        
        # ============
        # 1.1 Embedding layers for users and items
        # ============
        # These layers convert user and item IDs into dense vectors of size embedding_size.

        self.user_embedding_gmf = nn.Embedding(num_users, embedding_size)  # For GMF branch
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_size)  # For GMF branch

        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size)  # For MLP branch
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_size)  # For MLP branch
        
        # ============
        # 1.2 MLP branch: Multi-layer perceptron
        # ============
        # Define a sequence of fully connected layers for the MLP branch.

        mlp_modules = []
        input_size = embedding_size * 2
        for output_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.ReLU())  # Activation function 
            input_size = output_size
        self.mlp = nn.Sequential(*mlp_modules)
        
        # ============
        # 1.3 Fusion layer: Combine GMF and MLP outputs
        # ============
        # Combine the GMF output and MLP output.

        self.fusion_layer = nn.Linear(embedding_size + mlp_layers[-1], 1)
        
        # ============
        # 1.4 Final prediction layer: Sigmoid activation
        # ============
        # Sigmoid converts the output to a probability (0 to 1) for binary classification.

        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        # ============
        # 1.5 GMF branch: Generalized Matrix Factorization
        # ============
    
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_emb_gmf * item_emb_gmf  
        
        # ============
        # 1.6 MLP branch: Multi-layer perceptron
        # ============

        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)  
        mlp_output = self.mlp(mlp_input)
        
        # ============
        # 1.7 Combine GMF and MLP outputs
        # ============
        # Concatenate GMF and MLP outputs, then pass through the fusion layer.
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        fusion_output = self.fusion_layer(combined)
        
        # ============
        # 1.8 Final prediction
        # ============
        # Apply sigmoid to get a probability of interaction (0 to 1).
        prediction = self.sigmoid(fusion_output)
        return prediction

# ============
# 2. Load preprocessed data and prepare for training
# ============
# This function loads the CSV files and extracts user IDs, item IDs, and labels.

def load_preprocessed_data(train_file, val_file, test_file):
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)
    
    # Extract inputs (userId, movieId) and labels
    train_users = torch.tensor(train_data['userId'].values, dtype=torch.long)
    train_items = torch.tensor(train_data['movieId'].values, dtype=torch.long)
    train_labels = torch.tensor(train_data['label'].values, dtype=torch.float32)
    
    val_users = torch.tensor(val_data['userId'].values, dtype=torch.long)
    val_items = torch.tensor(val_data['movieId'].values, dtype=torch.long)
    val_labels = torch.tensor(val_data['label'].values, dtype=torch.float32)
    
    test_users = torch.tensor(test_data['userId'].values, dtype=torch.long)
    test_items = torch.tensor(test_data['movieId'].values, dtype=torch.long)
    test_labels = torch.tensor(test_data['label'].values, dtype=torch.float32)
    
    return (train_users, train_items, train_labels), (val_users, val_items, val_labels), (test_users, test_items, test_labels)

# ============
# 3. Training function to optimize the NCF model
# ============
# This function implements the training loop with binary cross-entropy loss,
# Adam optimizer, and early stopping based on validation loss.
def train_model(model, train_data, val_data, epochs=50, batch_size=256, lr=0.0005, patience=5):
    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss for binary labels (0 or 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
    
    # Unpack data
    train_users, train_items, train_labels = train_data
    val_users, val_items, val_labels = val_data
    
    # Variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        num_batches = (len(train_users) + batch_size - 1) // batch_size  # Ceiling division
        
        # Process data in batches
        for i in range(0, len(train_users), batch_size):
            # Get batch
            batch_users = train_users[i:i + batch_size]
            batch_items = train_items[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]
            
            # Forward pass
            optimizer.zero_grad()  # Clear previous gradients
            predictions = model(batch_users, batch_items).squeeze()  # Remove extra dimension
            loss = criterion(predictions, batch_labels)
            
            # Backward pass and optimization
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            total_loss += loss.item()
        
        # Average training loss
        train_loss = total_loss / num_batches
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No gradient computation
            val_predictions = model(val_users, val_items).squeeze()
            val_loss = criterion(val_predictions, val_labels).item()
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter if we improve
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

# ============
# 4. Evaluation function to compute Recall@10 and NDCG@10
# ============
# This function evaluates the trained model on the test set using ranking metrics.
def evaluate_model(model, test_data, num_items, k=10):
    model.eval()  # Set model to evaluation mode
    test_users, test_items, test_labels = test_data
    
    # Get unique users in the test set
    unique_users = torch.unique(test_users)
    recall_sum = 0.0
    ndcg_sum = 0.0
    
    with torch.no_grad():  # No gradient computation
        for user in unique_users:
            # Get all test data for this user
            user_mask = test_users == user
            user_items = test_items[user_mask]
            user_labels = test_labels[user_mask]
            
            # Generate predictions for all possible items
            all_items = torch.arange(num_items, dtype=torch.long)
            user_tensor = user.repeat(num_items)  # Repeat user ID for each item
            predictions = model(user_tensor, all_items).squeeze()
            
            # Get top-k predicted items
            _, top_k_indices = torch.topk(predictions, k)
            top_k_items = all_items[top_k_indices]
            
            # Relevant items (true positives)
            relevant_items = user_items[user_labels == 1]
            num_relevant = len(relevant_items)
            
            if num_relevant == 0:  # Skip if no relevant items
                continue
            
            # Recall@10: Fraction of relevant items in top-k
            hits = sum(item in relevant_items for item in top_k_items)
            recall = hits / min(num_relevant, k)  # Cap denominator at k
            recall_sum += recall
            
            # NDCG@10: Normalized Discounted Cumulative Gain
            dcg = 0.0
            for i, item in enumerate(top_k_items):
                if item in relevant_items:
                    dcg += 1 / np.log2(i + 2)  # i+2 because i is 0-based
            idcg = sum(1 / np.log2(i + 2) for i in range(min(num_relevant, k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_sum += ndcg
    
    # Average over all users
    num_users = len(unique_users)
    avg_recall = recall_sum / num_users
    avg_ndcg = ndcg_sum / num_users
    return avg_recall, avg_ndcg

# ============
# 5. Comparison function to evaluate different model configurations in two stages
# ============
# This function performs a two-stage comparison: first varying MLP layers, then embedding sizes,
# and plots results in a collaged graph.
def compare_models(train_data, val_data, test_data, num_users, num_items):
    # Stage 1: Vary MLP layers with fixed embedding size
    mlp_configs = [
        {"mlp_layers": [32, 16], "embedding_size": 32, "name": "MLP [32, 16]"},
        {"mlp_layers": [64, 32, 16], "embedding_size": 32, "name": "MLP [64, 32, 16]"},
        {"mlp_layers": [128, 64, 32], "embedding_size": 32, "name": "MLP [128, 64, 32]"}
    ]
    
    mlp_results = []
    print("\nStage 1: Comparing MLP configurations (Embedding Size = 32)")
    for config in mlp_configs:
        print(f"\nTraining model: {config['name']}")
        model = NCF(num_users, num_items, embedding_size=config["embedding_size"], mlp_layers=config["mlp_layers"])
        train_model(model, train_data, val_data)
        recall, ndcg = evaluate_model(model, test_data, num_items)
        mlp_results.append({"name": config["name"], "recall": recall, "ndcg": ndcg})
        print(f"{config['name']} - Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")
    
    # Select best MLP based on excelling in both Recall@10 and NDCG@10
    best_mlp = max(mlp_results, key=lambda x: (x["recall"], x["ndcg"]))  # Highest Recall@10, then NDCG@10
    print(f"\nBest MLP configuration: {best_mlp['name']} (Recall@10: {best_mlp['recall']:.4f}, NDCG@10: {best_mlp['ndcg']:.4f})")
    best_mlp_layers = [config["mlp_layers"] for config in mlp_configs if config["name"] == best_mlp["name"]][0]
    
    # Stage 2: Vary embedding size with best MLP
    embed_configs = [
        {"mlp_layers": best_mlp_layers, "embedding_size": 16, "name": "Embed 16"},
        {"mlp_layers": best_mlp_layers, "embedding_size": 32, "name": "Embed 32"},
        {"mlp_layers": best_mlp_layers, "embedding_size": 64, "name": "Embed 64"}
    ]
    
    embed_results = []
    print(f"\nStage 2: Comparing embedding sizes (MLP = {best_mlp['name']})")
    for config in embed_configs:
        print(f"\nTraining model: {config['name']}")
        model = NCF(num_users, num_items, embedding_size=config["embedding_size"], mlp_layers=config["mlp_layers"])
        train_model(model, train_data, val_data)
        recall, ndcg = evaluate_model(model, test_data, num_items)
        embed_results.append({"name": config["name"], "recall": recall, "ndcg": ndcg})
        print(f"{config['name']} - Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")
    
    # Plotting: Collage two graphs side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Graph 1: MLP comparison
    mlp_names = [r["name"] for r in mlp_results]
    mlp_recalls = [r["recall"] for r in mlp_results]
    mlp_ndcgs = [r["ndcg"] for r in mlp_results]
    bar_width = 0.35
    x = np.arange(len(mlp_names))
    bars1 = ax1.bar(x - bar_width/2, mlp_recalls, bar_width, label="Recall@10", color="skyblue")
    bars2 = ax1.bar(x + bar_width/2, mlp_ndcgs, bar_width, label="NDCG@10", color="lightgreen")
    ax1.set_title("Performance Across MLP Configurations\n(Embed = 32)")
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 0.1)  
    ax1.set_xticks(x)
    ax1.set_xticklabels(mlp_names, rotation=15)
    ax1.legend(loc="upper right")
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 0.9, f"{height:.4f}", ha="center", va="top")
    
    # Graph 2: Embedding size comparison
    embed_names = [r["name"] for r in embed_results]
    embed_recalls = [r["recall"] for r in embed_results]
    embed_ndcgs = [r["ndcg"] for r in embed_results]
    x = np.arange(len(embed_names))
    bars1 = ax2.bar(x - bar_width/2, embed_recalls, bar_width, label="Recall@10", color="skyblue")
    bars2 = ax2.bar(x + bar_width/2, embed_ndcgs, bar_width, label="NDCG@10", color="lightgreen")
    ax2.set_title(f"Performance Across Embedding Sizes\n(MLP = {best_mlp['name']})")
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 0.1)  
    ax2.set_xticks(x)
    ax2.set_xticklabels(embed_names, rotation=15)
    ax2.legend(loc="upper right")
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 0.9, f"{height:.4f}", ha="center", va="top")
    
    # Finalize and save collage
    plt.tight_layout()
    plt.savefig("comparison_collage.png")
    plt.close()
    
    # Combine results for table
    all_results = mlp_results + embed_results
    return all_results

# ============
# 6. Main function to set up and compare the model configurations
# ============

def main():
    # Load preprocessed data
    train_data, val_data, test_data = load_preprocessed_data("train_data.csv", "val_data.csv", "test_data.csv")
    train_users, train_items, _ = train_data
    
    # Get the number of unique users and items
    num_users = max(train_users.max(), val_data[0].max(), test_data[0].max()) + 1
    num_items = max(train_items.max(), val_data[1].max(), test_data[1].max()) + 1
    
    print("Model setup with", num_users, "users and", num_items, "items.")
    
    # Compare configurations in two stages
    print("\nStarting two-stage model comparison...")
    results = compare_models(train_data, val_data, test_data, num_users, num_items)
    
    # Print comparison table
    print("\nComparison Table:")
    print("Configuration                | Recall@10 | NDCG@10")
    print("-----------------------------|-----------|--------")
    for result in results:
        print(f"{result['name']:<28} | {result['recall']:.4f}   | {result['ndcg']:.4f}")

if __name__ == "__main__":
    main()