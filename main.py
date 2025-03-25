import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import heapq


# Configuration
DATA_PATH = 'ml-1m/ml-1m/ratings.dat'  # Update with the actual path
NUM_NEGATIVE_SAMPLES = 4
EMBEDDING_DIM = 32
MLP_LAYERS = [64, 32, 16, 8]  # Adjusted MLP layers for better performance
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
TOP_K = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Data loading and pre-processing

def load_data(data_path):
    attr = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(data_path, sep='::', header=None, names=attr)
    return df

def preprocess_data(df):
    df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    df['user_id'] = user_enc.fit_transform(df['user_id'])
    df['item_id'] = item_enc.fit_transform(df['item_id'])

    users = len(user_enc.classes_)
    items = len(item_enc.classes_)

    return df, users, items

def generate_negative_samples(df, users, items, negatives):
    user_item_set = set(zip(df['user_id'], df['item_id']))
    negative_samples = []
    for u, i in tqdm(user_item_set, desc="Generating Negative Samples"):
        for _ in range(negatives):
            # Generate a random item that the user hasn't interacted with
            j = np.random.randint(items)
            while(u, j) in user_item_set:
                j = np.random.randint(items)
            negative_samples.append([u, j, 0])
    negative_df = pd.DataFrame(negative_samples, columns=['user_id', 'item_id', 'label'])
    return negative_df

class NCFDataset(Dataset):
    def __init__(self, df):
        self.user_id = df['user_id'].values
        self.item_id = df['item_id'].values
        self.labels = df['label'].values
    
    def __len__(self):
        return len(self.user_id)
    
    def __getitem__(self, idx):
        return(
            torch.tensor(self.user_id[idx], dtype=torch.long),
            torch.tensor(self.item_id[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )

class NCF(nn.Module):
    def __init__(self, users, items, emb_dimenstion, mlp_layers):
        super(NCF, self).__init__()
        self.user_embedding_gmf = nn.Embedding(users, emb_dimenstion)
        self.item_embedding_gmf = nn.Embedding(items, emb_dimenstion)
        self.user_embedding_mlp = nn.Embedding(users, emb_dimenstion)
        self.item_embedding_mlp = nn.Embedding(items, emb_dimenstion)

        # gmf
        self.gmf_linear = nn.Linear(emb_dimenstion, mlp_layers[-1])

        # mlp
        mlp_layer_sizes = [2 * emb_dimenstion] + mlp_layers
        self.mlp_layers = nn.ModuleList()
        for i in range(len(mlp_layer_sizes - 1)):
            self.mlp_layers.append(nn.Linear(mlp_layer_sizes[i], mlp_layer_sizes[i + 1]))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.BatchNorm1d(mlp_layer_sizes[i + 1]))
        
        # Futione Layer
        self.fusion_layer = nn.Linear(mlp_layers[-1] + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.fusion_layer.weight)
    
    def forward(self, user_ids, item_ids):

        # GMF

        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_emb_gmf * item_emb_gmf
        gmf_output = self.gmf_linear(gmf_output)

        # MLP

        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat((user_emb_mlp, item_emb_mlp), dim=1)
        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
        
        # Fusion
        fusion_input = torch.cat((gmf_output, mlp_output), dim=-1)
        output = self.fusion_layer(fusion_input)
        output = self.sigmoid(output)
        return output.squeeze()
    
# Evaluation Metrics
def hit_ratio(predicted, ground_truth):
    return 1 if ground_truth in predicted else 0

def ndcg(predicted, ground_truth):
    if ground_truth in predicted:
        index = predicted.index(ground_truth)
        return 1 / np.log2(index + 2)
    return 0

def evaluate(model, data_loader, top_k):
    model.eval()
    hr_sum, ndcg_sum = 0, 0
    num_users = 0

    with torch.no_grad():
        user_item_pairs = {}
        for user, item, label in data_loader.dataset:
            user_id = user.item()
            item_id = item.item()
            if user_id not in user_item_pairs:
                user_item_pairs[user_id] = []
            user_item_pairs[user_id].append((item_id, label.item()))
        
        for user_id, items in user_item_pairs.item():
            pos_items = [item_id for item_id, label in items if label == 1]
            if not pos_items:
                continue
            
            # Get all item Ids for this user
            all_items = [item_id for item_id, _ in items]

            # Get predictions for all items the user interacted with
            predictions = model(torch.tensor([user_id] * len(all_items)).to(DEVICE), torch.tensor(all_items).to(DEVICE))

            # Sort the prediction and get top-k item indices
            _, indices = torch.topk(predictions, top_k)
            recommended_items = [all_items[i] for i in indices.cpu().numpy()]

            # Calculate HR and NDCG for each positive item
            for pos_item in pos_items:
                hr_sum += hit_ratio(recommended_items, pos_item)
                ndcg_sum += ndcg(recommended_items, pos_item)
            
            num_users += len(pos_items) # Count each positive item as a user instance
    
    hr = hr_sum / num_users if num_users > 0 else 0
    ndcg_ = ndcg_sum / num_users if num_users > 0 else 0

    return hr, ndcg_

# Training Loop
def train(model, train_loader, val_loader, optimizer, epochs, early_stopping_patience):
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for user_ids, item_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            user_ids = user_ids.to(DEVICE)
            item_ids = item_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zro_grad()
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for user_ids, item_ids, label in val_loader:
                user_ids = user_ids.to(DEVICE)
                item_ids = item_ids.to(DEVICE)
                label = labels.to(DEVICE)
                outputs = model(user_ids, item_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss: .4f}, Validation Loss: {avg_val_loss: .4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth') # Save best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print("Early stopping triggered.")
                break
    print("Finished Training")

if __name__ == "__main__":

    # 1. Load and Preprocess Data

    df = load_data(DATA_PATH)
    df, num_users, num_items = preprocess_data(df)

    #2. Generate Negative Samples

    negative_df = generate_negative_samples(df, num_users, num_items, NUM_NEGATIVE_SAMPLES)
    df = pd.concat([df[df['label'] == 1], negative_df], ignore_index=True) # Only keep positive interactions

    # 3. Split Data

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=RANDOM_SEED)

    # 4. Create Datasets and DataLoader
    train_dataset = NCFDataset(train_df)
    val_dataset = NCFDataset(val_df)
    test_dataset = NCFDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 5. Initialize Model, Optimizer, and Train
    model = NCF(num_users, num_items, EMBEDDING_DIM, MLP_LAYERS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, val_loader, optimizer, EPOCHS, EARLY_STOPPING_PATIENCE)

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(DEVICE)

    # 6. Evaluate Model
    hr, ndcg_score = evaluate(model, test_loader, TOP_K)
    print(f"Recall@{TOP_K}: {hr: .4f}")
    print(f"NDCG@{TOP_K}: {ndcg_score: .4f}")