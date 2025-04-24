# --- Constants ---
MAX_LEN = 200         # Maximum sequence length for BERT4Rec (Requirement: fixed length, e.g., 20 - adjust if needed)
# The paper mentions using N=200 for ML-1M, N=50 for Beauty/Steam.
# Let's use 50 as a reasonable default, adjustable if needed.
MASK_PROB = 0.15     # Probability of masking an item in the sequence for training (Paper explored 0.1-0.9, found 0.2 optimal for ML-1M, 0.4 for Steam, 0.6 for Beauty)
# We'll use 0.2 as a default, can be made configurable if needed per dataset.
# Model Hyperparameters (These will be varied in configurations)
# BERT_HIDDEN_DIM = 128 # Defined in configs
# BERT_ATTENTION_HEADS = 4 # Defined in configs
# BERT_LAYERS = 2 # Defined in configs
BERT_DROPOUT = 0.2 # Kept constant for simplicity, but could be varied (Paper likely used 0.1 or 0.2 based on BERT practices)
# Training Hyperparameters
TRAIN_BATCH_SIZE = 128 # Paper used 256, smaller batch size might be needed depending on GPU memory
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 200 # Max epochs, early stopping will likely trigger sooner (Paper doesn't specify epochs, but likely ran until convergence/early stopping)
PATIENCE = 10 # Patience for early stopping based on Val NDCG@10 (Increased patience slightly)
TOP_K = 10 # For Recall@K and NDCG@K
# SCHEDULER_STEP_SIZE = 3 # Paper mentions linear decay, not step decay. Let's remove StepLR and implement linear decay or use AdamW.
# SCHEDULER_GAMMA = 0.1
# AdamW is often preferred for Transformer models
WEIGHT_DECAY = 0.01 # Paper mentions l2 weight decay of 0.01