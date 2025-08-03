import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
import os
from models.model import TabularMLP
from utils.data_utils import process_data
from utils.metrics import binary_accuracy, precision_recall_f1

# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------
# Load all evaluation, model, and data setup from the YAML config file.
# This ensures reproducibility and a single source of truth for experiment settings.
with open('./configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
# 1. Preprocess data and load validation split.
#    - Features are scaled and encoded per training setup.
#    - Ensures that evaluation uses the same preprocessing steps as training.
X_train, y_train, X_val, y_val, n_in = process_data(
    csv_path=cfg['dataset']['csv_path'],
    test_size=cfg['dataset']['test_size'],
    random_state=cfg['dataset']['random_state']
)

# 2. Wrap validation data as a PyTorch DataLoader for efficient mini-batch evaluation.
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=cfg['training']['batch_size']
)

# ---------------------------------------------------------------------------
# Model Setup & Loading
# ---------------------------------------------------------------------------
# 1. Select computation device (GPU if available, otherwise CPU)
device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')

# 2. Build the tabular MLP model instance matching training configuration.
model = TabularMLP(
    in_features=n_in,
    hidden_dims=cfg['model']['hidden_dims'],
    dropout=cfg['model']['dropout']
).to(device)

# 3. Find latest model checkpoint file (saved state from training).
checkpoint_dir = cfg['logging']['checkpoint_dir']
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
latest_ckpt = sorted(checkpoints)[-1]  # Assumes filenames are sortable by epoch

# 4. Load model weights (state_dict) from latest checkpoint.
state = torch.load(os.path.join(checkpoint_dir, latest_ckpt))
model.load_state_dict(state['model_state_dict'])
model.eval()  # Set model to evaluation mode (disables dropout, enables batchnorm inference)

# ---------------------------------------------------------------------------
# Evaluation Loop (Validation)
# ---------------------------------------------------------------------------
# Initialize accumulators for metrics
val_loss, val_acc, val_f1 = 0, 0, 0
criterion = torch.nn.BCEWithLogitsLoss()  # Suitable for binary classification logits

# Do not track gradients during evaluation (reduces memory, speeds up)
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                         # Forward pass to get output logits
        loss = criterion(logits, yb)               # Compute batch loss
        val_loss += loss.item() * xb.size(0)       # Accumulate loss, scaled by batch size
        val_acc += binary_accuracy(logits, yb) * xb.size(0)  # Accumulate accuracy
        _, _, f1 = precision_recall_f1(logits, yb)           # Get F1-score for batch
        val_f1 += f1 * xb.size(0)                  # Accumulate F1, scaled by batch size

# Divide weighted sums by number of samples to get final averages
val_loss /= len(X_val)
val_acc /= len(X_val)
val_f1 /= len(X_val)

# ---------------------------------------------------------------------------
# Results Output
# ---------------------------------------------------------------------------
# Print summary of validation performance metrics
print(f"Validation Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}")
