import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import yaml
import os

from models.model import TabularMLP
from utils.data_utils import process_data
from utils.metrics import binary_accuracy, precision_recall_f1
from utils.logger import setup_logger

# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------
# Load training, model, and data configuration from YAML file for reproducibility
with open('./configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
# Initialize logger to track experiment metrics, status messages, and diagnostics
logger = setup_logger(cfg['logging']['log_dir'])

# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
# 1. Load and preprocess dataset using custom utility.
#    - Applies one-hot encoding, scaling, missing value filtration, and splitting.
#    - Also persists transformer objects for later inference use.
X_train, y_train, X_val, y_val, n_in = process_data(
    csv_path=cfg['dataset']['csv_path'],
    test_size=cfg['dataset']['test_size'],
    random_state=cfg['dataset']['random_state'],
    save_dir=cfg['logging']['log_dir']
)

# 2. Wrap numpy arrays into PyTorch TensorDatasets and DataLoaders for efficient batch processing
train_loader = DataLoader(
    TensorDataset(X_train, y_train), 
    batch_size=cfg['training']['batch_size'], 
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val), 
    batch_size=cfg['training']['batch_size']
)

# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------
# 1. Device configuration: Use GPU if available, otherwise CPU
device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')

# 2. Instantiate the MLP model with parameters from config file
model = TabularMLP(
    in_features=n_in,
    hidden_dims=cfg['model']['hidden_dims'],
    dropout=cfg['model']['dropout']
).to(device)

# 3. Log the model architecture for reference
logger.info(f"Model: {model}")

# 4. Setup binary classification loss and Adam optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
# Iterate over the specified number of epochs for training/validation
for epoch in range(1, cfg['training']['epochs'] + 1):
    # Set the model to training mode (enables dropout/batchnorm, etc.)
    model.train()
    train_loss, train_acc = 0, 0

    # -----------------------------
    # Training Phase (per epoch)
    # -----------------------------
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()                # Reset gradients
        logits = model(xb)                   # Forward pass
        loss = criterion(logits, yb)         # Compute loss
        loss.backward()                      # Backward pass (gradient computation)
        optimizer.step()                     # Update weights
        
        # Running accumulators (sum loss and accuracy * batch_size for averaging)
        train_loss += loss.item() * xb.size(0)
        train_acc += binary_accuracy(logits, yb) * xb.size(0)
    
    # Compute average training loss and accuracy
    train_loss /= len(X_train)
    train_acc /= len(X_train)

    # -----------------------------
    # Validation Phase (per epoch)
    # -----------------------------
    model.eval()                             # Set model to inference mode (disables dropout)
    val_loss, val_acc, val_f1 = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            val_acc += binary_accuracy(logits, yb) * xb.size(0)
            _, _, f1 = precision_recall_f1(logits, yb)
            val_f1 += f1 * xb.size(0)
    # Compute average
    val_loss /= len(X_val)
    val_acc /= len(X_val)
    val_f1 /= len(X_val)

    # Log metrics to both console and file via logger
    logger.info(
        f"Epoch {epoch}: "
        f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
        f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}"
    )

    # -----------------------------
    # Model Checkpointing
    # -----------------------------
    # Save model and optimizer state for recovery, future evaluation, or deployment
    os.makedirs(cfg['logging']['checkpoint_dir'], exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        },
        os.path.join(
            cfg['logging']['checkpoint_dir'],
            f'model_epoch_{epoch}.pth'
        )
    )
