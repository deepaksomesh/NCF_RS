# Neural Collaborative Filtering for Movie Recommendations

This project focuses on implementing and experimenting with **Neural Collaborative Filtering (NCF)** for movie recommendations using the **MovieLens 1M** dataset.

## 📚 Overview

We preprocess the MovieLens dataset by:

- Converting ratings into **binary implicit feedback**
- Performing **negative sampling**
- Splitting the data into **training, validation, and test** sets

Three NCF models are designed:

- **Small**
- **Medium**
- **Large**

Each model combines:
- **Generalized Matrix Factorization (GMF)**
- **Multi-Layer Perceptron (MLP)** branches

The models are trained for **5 epochs** with **early stopping**, achieving **NDCG@10 scores above 0.58**.

### 🔍 Evaluation

Models are evaluated using:
- **Recall@10**
- **NDCG@10**

We compare their performance, draw insights, and discuss potential improvements. While NCF proves effective for movie recommendations, there is still room to enhance **Recall@10**.

---

## ⚙️ Setup & Usage

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```
### ⬇️ Download the Dataset
Download the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/), extract it, and place the folder in the same directory as your python files

### 🗃️ Prepare the Dataset
From the extracted `ml-1m` folder, copy the `ratings.dat` file and place it in the **same directory** as your `ncf.py` script.  
This ensures that the script can access the dataset without any file path issues.

### Run the Model
```bash
python ncf.py
```
Sit back and watch the magic happen ✨
