import torch  
import torch.nn as nn 
import torch.optim as optim  
import pandas as pd  
import numpy as np  
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt  
from tqdm import tqdm  

# Step 1: Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Step 2: Define function to load and preprocess the MovieLens 1M dataset
def load_and_preprocess_data():
    """
    Loads the MovieLens 1M dataset, converts ratings to binary implicit feedback,
    performs negative sampling, and splits the data into train/val/test sets.
    
    Returns:
        train (DataFrame): Training data with userId, movieId, and label.
        val (DataFrame): Validation data.
        test (DataFrame): Test data.
        val_pos (DataFrame): Positive interactions for validation users.
        test_pos (DataFrame): Positive interactions for test users.
        num_users (int): Total number of users.
        num_items (int): Total number of items.
    """
    # Step 2.1: Load the dataset
    print("Loading MovieLens 1M dataset...")
    ratings = pd.read_csv("ratings.dat", sep="::", engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
    
    # Step 2.2: Convert ratings to binary implicit feedback
    print("Converting ratings to binary implicit feedback (ratings >= 4 as positive)...")
    ratings['label'] = (ratings['rating'] >= 4).astype(int)
    
    # Step 2.3: Store positive interactions for evaluation
    positive_interactions = ratings[ratings['label'] == 1][['userId', 'movieId']].copy()
    
    # Step 2.4: Perform negative sampling (4 negatives per positive)
    print("Performing negative sampling (4 negatives per positive)...")
    user_item_set = set(zip(ratings['userId'], ratings['movieId']))  # Set of (user, item) pairs
    all_movie_ids = ratings['movieId'].unique()  # All unique movie IDs
    negative_samples = []
    num_negatives = 4
    
    for user in ratings['userId'].unique():
        interacted_movies = ratings.loc[ratings['userId'] == user, 'movieId'].tolist()
        for _ in range(len(interacted_movies) * num_negatives):
            negative_movie = np.random.choice(all_movie_ids)
            while (user, negative_movie) in user_item_set:
                negative_movie = np.random.choice(all_movie_ids)
            negative_samples.append([user, negative_movie, 0])
    
    # Step 2.5: Combine positive and negative samples
    neg_samples = pd.DataFrame(negative_samples, columns=['userId', 'movieId', 'label'])
    dataset = pd.concat([ratings[['userId', 'movieId', 'label']], neg_samples])
    
    # Step 2.6: Split the dataset into train/val/test sets
    print("Splitting dataset: 70% training, 15% validation, 15% testing...")
    train, temp = train_test_split(dataset, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    # Step 2.7: Ensure validation and test sets have positive interactions
    val_users = set(val['userId'].unique())
    test_users = set(test['userId'].unique())
    val_pos = positive_interactions[positive_interactions['userId'].isin(val_users)]
    test_pos = positive_interactions[positive_interactions['userId'].isin(test_users)]
    
    return train, val, test, val_pos, test_pos, ratings['userId'].max() + 1, ratings['movieId'].max() + 1

# Step 3: Define custom Dataset class for NCF
class NCFDataset(Dataset):
    """
    A custom PyTorch Dataset class for NCF, supporting both training and evaluation modes.
    
    Args:
        data (DataFrame): DataFrame with userId, movieId, and label.
        positive_data (DataFrame, optional): Positive interactions for evaluation.
        all_items (Tensor, optional): Tensor of all item indices for evaluation.
        is_eval (bool): If True, dataset is used for evaluation.
    """
    def __init__(self, data, positive_data=None, all_items=None, is_eval=False):
        # Step 3.1: Initialize dataset parameters
        self.data = data
        self.positive_data = positive_data
        self.is_eval = is_eval
        self.all_items = all_items
        if is_eval:
            # Step 3.2: Store unique users for evaluation mode
            self.unique_users = np.unique(data['userId'].values)
        else:
            # Step 3.3: Store user, item, and label tensors for training mode
            self.users = torch.tensor(data['userId'].values, dtype=torch.long)
            self.items = torch.tensor(data['movieId'].values, dtype=torch.long)
            self.labels = torch.tensor(data['label'].values, dtype=torch.float32)
    
    def __len__(self):
        # Step 3.4: Return the number of samples
        if self.is_eval:
            return len(self.unique_users)
        return len(self.users)
    
    def __getitem__(self, idx):
        # Step 3.5: Return a single sample based on mode
        if self.is_eval:
            user = torch.tensor(self.unique_users[idx], dtype=torch.long)
            items = self.all_items
            return user, items
        else:
            return self.users[idx], self.items[idx], self.labels[idx]

# Step 4: Define the NCF model
class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model combining Generalized Matrix Factorization (GMF)
    and Multi-Layer Perceptron (MLP) branches.
    
    Args:
        num_users (int): Number of users.
        num_items (int): Number of items.
        gmf_emb_dim (int): Embedding dimension for GMF.
        mlp_emb_dim (int): Embedding dimension for MLP.
        mlp_layers (list): List of layer sizes for the MLP branch.
    """
    def __init__(self, num_users, num_items, gmf_emb_dim=32, mlp_emb_dim=32, mlp_layers=[64, 32]):
        super().__init__()
        # Step 4.1: Define GMF embeddings
        self.gmf_user_emb = nn.Embedding(num_users, gmf_emb_dim)
        self.gmf_item_emb = nn.Embedding(num_items, gmf_emb_dim)
        
        # Step 4.2: Define MLP embeddings
        self.mlp_user_emb = nn.Embedding(num_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, mlp_emb_dim)
        # Step 4.3: Build MLP layers with ReLU and Dropout
        mlp_modules = []
        in_dim = mlp_emb_dim * 2
        for out_dim in mlp_layers:
            mlp_modules.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_modules)
        
        # Step 4.4: Define final fusion layer
        self.final = nn.Linear(gmf_emb_dim + mlp_layers[-1], 1)
        
        # Step 4.5: Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Step 4.6: Initialize embeddings and linear layers with Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user, item):
        # Step 4.7: Compute GMF path
        gmf_user = self.gmf_user_emb(user)
        gmf_item = self.gmf_item_emb(item)
        gmf_out = gmf_user * gmf_item
        
        # Step 4.8: Compute MLP path
        mlp_user = self.mlp_user_emb(user)
        mlp_item = self.mlp_item_emb(item)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        
        # Step 4.9: Handle shapes for training and evaluation
        if len(user.shape) == 1:
            mlp_out = self.mlp(mlp_input)
        else:
            batch_size, num_items, _ = mlp_input.shape
            mlp_input = mlp_input.view(-1, mlp_input.shape[-1])
            mlp_out = self.mlp(mlp_input)
            mlp_out = mlp_out.view(batch_size, num_items, -1)
            gmf_out = gmf_out.view(batch_size, num_items, -1)
        
        # Step 4.10: Combine GMF and MLP outputs
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        final_out = self.final(combined)
        
        # Step 4.11: Squeeze the output dimension
        if len(final_out.shape) == 3:
            final_out = final_out.squeeze(-1)
        else:
            final_out = final_out.squeeze(-1)
        
        return final_out

# Step 5: Define training function
def train_model(model, train_loader, val_loader, val_metric_loader, num_items, device, epochs=5, patience=3, lr=0.001):
    """
    Trains the NCF model with early stopping based on validation loss.
    
    Args:
        model (nn.Module): NCF model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data (loss computation).
        val_metric_loader (DataLoader): DataLoader for validation data (metric computation).
        num_items (int): Total number of items.
        device (torch.device): Device to run the model on.
        epochs (int): Number of epochs to train.
        patience (int): Number of epochs to wait for improvement before early stopping.
        lr (float): Learning rate for the Adam optimizer.
    
    Returns:
        best_val_loss (float): Best validation loss achieved.
        model (nn.Module): Trained model with the best weights.
    """
    # Step 5.1: Initialize training parameters
    print(f"\n=== Training NCF ===")
    print(f"Starting training for NCF (LR={lr}, patience={patience})...\n")
    pos_weight = torch.tensor([1.5]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Step 5.2: Training loop over epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        
        # Step 5.3: Train the model on the training data
        train_progress = tqdm(train_loader, total=len(train_loader), desc="Training")
        for users, items, labels in train_progress:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(users, items)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")
        
        # Step 5.4: Compute validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_loss_progress = tqdm(val_loader, total=len(val_loader), desc="Computing Validation Loss")
            for users, items, labels in val_loss_progress:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                outputs = model(users, items)
                val_loss += criterion(outputs, labels).item()
        
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Step 5.5: Compute validation metrics
        val_recall, val_ndcg = evaluate_model(model, val_metric_loader, num_items, device, sample_items=1000)
        print(f"Val Recall@10: {val_recall:.4f}")
        print(f"Val NDCG@10: {val_ndcg:.4f}")
        
        # Step 5.6: Implement early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            print("New best model saved!\n")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss, epochs without improvement: {epochs_no_improve}/{patience}\n")
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
        
        scheduler.step()
    
    # Step 5.7: Load the best model weights
    model.load_state_dict(best_model_state)
    print("Training completed, best model state loaded.")
    return best_val_loss, model

# Step 6: Define evaluation function
def evaluate_model(model, data_loader, num_items, device, sample_items=1000):
    """
    Evaluates the model by computing Recall@10 and NDCG@10 for each user.
    
    Args:
        model (nn.Module): Trained NCF model.
        data_loader (DataLoader): DataLoader for evaluation data.
        num_items (int): Total number of items.
        device (torch.device): Device to run the model on.
        sample_items (int): Number of items to sample for evaluation (use all items if equal to num_items).
    
    Returns:
        recall (float): Average Recall@10 across all users.
        ndcg (float): Average NDCG@10 across all users.
    """
    # Step 6.1: Set model to evaluation mode
    model.eval()
    recall_list = []
    ndcg_list = []
    
    # Step 6.2: Evaluate each user
    eval_progress = tqdm(data_loader, total=len(data_loader), desc="Evaluating")
    with torch.no_grad():
        for i, (user, items) in enumerate(eval_progress):
            user = user.to(device)
            # Step 6.3: Sample items for evaluation
            if sample_items < num_items:
                item_indices = torch.randperm(num_items)[:sample_items].to(device)
            else:
                item_indices = torch.arange(num_items).to(device)
            users_expanded = user.unsqueeze(0).expand(1, len(item_indices))
            items_sampled = item_indices.unsqueeze(0)
            outputs = model(users_expanded, items_sampled)
            outputs = torch.sigmoid(outputs)
            
            # Step 6.4: Convert predictions to numpy
            user_preds = outputs[0].cpu().numpy()
            user_labels = np.zeros(len(item_indices))
            user_id = user.cpu().item()
            # Step 6.5: Get ground truth labels
            user_data = data_loader.dataset.positive_data[
                data_loader.dataset.positive_data['userId'] == user_id
            ]
            interacted_items = user_data['movieId'].values
            item_indices_np = item_indices.cpu().numpy()
            for item in interacted_items:
                if item in item_indices_np:
                    idx = np.where(item_indices_np == item)[0][0]
                    user_labels[idx] = 1
            
            # Step 6.6: Compute Recall@10 and NDCG@10
            top_k = np.argsort(user_preds)[-10:]
            relevant = user_labels[top_k]
            recall = np.sum(relevant) / np.sum(user_labels) if np.sum(user_labels) > 0 else 0
            dcg = np.sum(relevant / np.log2(np.arange(2, 12)))
            idcg = np.sum(np.sort(user_labels)[-10:] / np.log2(np.arange(2, 12)))
            ndcg = dcg / idcg if idcg > 0 else 0
            recall_list.append(recall)
            ndcg_list.append(ndcg)
    
    return np.mean(recall_list), np.mean(ndcg_list)

# Step 7: Define main function
def main():
    """
    Main function to load data, train NCF models, evaluate them, and save results.
    Trains three variants of NCF (Small, Medium, High) and compares their performance.
    """
    # Step 7.1: Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 7.2: Load and preprocess data
    print("Loading and preprocessing data...")
    train, val, test, val_pos, test_pos, num_users, num_items = load_and_preprocess_data()
    print(f"Dataset size: {len(train)} training, {len(val)} validation, {len(test)} test samples")
    print(f"Number of users: {num_users}, Number of items: {num_items}")
    
    # Step 7.3: Create DataLoaders
    all_items = torch.arange(num_items, dtype=torch.long)
    train_loader = DataLoader(NCFDataset(train), batch_size=512, shuffle=True)
    val_loader = DataLoader(NCFDataset(val), batch_size=512, shuffle=False)
    val_metric_loader = DataLoader(NCFDataset(val, positive_data=val_pos, all_items=all_items, is_eval=True), batch_size=1, shuffle=False)
    test_loader = DataLoader(NCFDataset(test, positive_data=test_pos, all_items=all_items, is_eval=True), batch_size=1, shuffle=False)
    
    # Step 7.4: Define model configurations
    model_configs = [
        {"gmf_emb_dim": 32, "mlp_emb_dim": 32, "mlp_layers": [64, 32], "name": "NCF_Small"},
        {"gmf_emb_dim": 32, "mlp_emb_dim": 32, "mlp_layers": [128, 64, 32], "name": "NCF_Medium"},
        {"gmf_emb_dim": 32, "mlp_emb_dim": 32, "mlp_layers": [256, 128, 64, 32], "name": "NCF_High"}
    ]
    
    # Step 7.5: Train and evaluate each model
    results = []
    for config in model_configs:
        print(f"\nTraining model: {config['name']}")
        print(f"GMF embedding dimension: {config['gmf_emb_dim']}, MLP embedding dimension: {config['mlp_emb_dim']}, MLP layers: {config['mlp_layers']}")
        model = NCF(num_users, num_items, config["gmf_emb_dim"], config["mlp_emb_dim"], config["mlp_layers"]).to(device)
        
        # Step 7.6: Train the model
        best_val_loss, model = train_model(model, train_loader, val_loader, val_metric_loader, num_items, device, epochs=5)
        
        # Step 7.7: Evaluate on the test set
        print("Evaluating on test set...")
        test_recall, test_ndcg = evaluate_model(model, test_loader, num_items, device, sample_items=num_items)
        print(f"Test Recall@10: {test_recall:.4f}")
        print(f"Test NDCG@10: {test_ndcg:.4f}")
        
        # Step 7.8: Store results
        results.append({
            "Model": config["name"],
            "MLP Layers": str(config["mlp_layers"]),
            "Test Recall@10": round(test_recall, 4),
            "Test NDCG@10": round(test_ndcg, 4)
        })
    
    # Step 7.9: Create comparison table and bar graph
    print("Creating model comparison table and bar graph...")
    df = pd.DataFrame(results)
    df.to_csv("model_comparison.csv", index=False)
    
    # Step 7.10: Plot the comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.axis("off")
    table = ax1.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    x = np.arange(len(results))
    width = 0.35
    ax2.bar(x - width/2, df["Test Recall@10"], width, label="Recall@10", color="#6BAED6")
    ax2.bar(x + width/2, df["Test NDCG@10"], width, label="NDCG@10", color="#FDBB84")
    for i, (recall, ndcg) in enumerate(zip(df["Test Recall@10"], df["Test NDCG@10"])):
        ax2.text(i - width/2, recall + 0.01, f"{recall:.4f}", ha="center", va="bottom", fontsize=10)
        ax2.text(i + width/2, ndcg + 0.01, f"{ndcg:.4f}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("Score")
    ax2.set_title("Model Performance Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Model"])
    ax2.set_ylim(0, 0.75)
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("model_comparison_plot.png")
    plt.close()
    print("Model comparison saved as model_comparison_plot.png and model_comparison.csv")

# Step 8: Run the main function
if __name__ == "__main__":
    main()