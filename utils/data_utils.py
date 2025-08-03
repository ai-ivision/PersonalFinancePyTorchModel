import os
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def detect_column_types(df, target_col, exclude_cols=None, unique_threshold=0.05):
    """
    Infer categorical and numeric columns automatically from a DataFrame.

    Args:
        df (pd.DataFrame): Input data.
        target_col (str): Name of the target column to exclude from features.
        exclude_cols (list or None): List of additional columns to exclude from features.
        unique_threshold (float): Threshold ratio to treat numeric columns with low unique count as categorical.

    Returns:
        categorical_cols (list of str): List of inferred categorical columns.
        numeric_cols (list of str): List of inferred numeric columns.
    """
    exclude_cols = exclude_cols or []
    feature_cols = [col for col in df.columns if col not in exclude_cols + [target_col]]

    categorical_cols = []
    numeric_cols = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df[col])
            if unique_ratio < unique_threshold:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Debug prints (comment out after you verify the data)
    print(f"Inferred categorical columns: {categorical_cols}")
    print(f"Inferred numeric columns: {numeric_cols}")

    return categorical_cols, numeric_cols


def process_data(csv_path, test_size, random_state, save_dir=None, target_col='has_loan'):
    """
    Loads data from CSV, preprocesses features and target, splits into train/val,
    and optionally saves preprocessing transformers.

    Args:
        csv_path (str): Path to the CSV data file.
        test_size (float): Fraction of data reserved for validation.
        random_state (int): Seed for train/val split reproducibility.
        save_dir (str or None): Directory to save fitted transformers for reuse.
        target_col (str): Name of the target column.

    Returns:
        X_train (torch.Tensor): Training features tensor (float32).
        y_train (torch.Tensor): Training target tensor (float32).
        X_val (torch.Tensor): Validation features tensor (float32).
        y_val (torch.Tensor): Validation target tensor (float32).
        num_features (int): Number of total features after preprocessing.
    """
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Initial data shape: {df.shape}")
    print(f"Columns in dataset: {df.columns.tolist()}")

    # Infer feature columns dynamically
    categorical_cols, numeric_cols = detect_column_types(df, target_col)

    # Handle missing values in categorical columns (drop rows with missing categoricals)
    if categorical_cols:
        df = df.dropna(subset=categorical_cols)
        print(f"Data shape after dropping missing categorical values: {df.shape}")

    # Drop rows with missing numeric features or missing target — important for consistency
    df = df.dropna(subset=numeric_cols + [target_col])
    print(f"Data shape after dropping missing numeric features & target: {df.shape}")

    # Process target column: convert to binary 0/1 integers safely
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].astype(str).str.strip().str.lower()
        # Map any form of yes/true/1 to 1 and no/false/0 to 0
        target_mapping = {
            'true': 1, 'false': 0,
            '1': 1, '0': 0,
            'yes': 1, 'no': 0,
            'y': 1, 'n': 0
        }
        df[target_col] = df[target_col].map(target_mapping)
        # Drop any rows where mapping yielded NaN (invalid target values or missing)
        df = df.dropna(subset=[target_col])
        df[target_col] = df[target_col].astype(int)
        # Debug counts
        print("[DEBUG] Target value counts after mapping:")
        print(df[target_col].value_counts())

    print(f"Target value counts after processing:\n{df[target_col].value_counts(dropna=False)}")
    print(f"Final dataset shape before encoding: {df.shape}")

    # One-Hot Encoding for categorical features (if any)
    enc = OneHotEncoder(handle_unknown='ignore')
    if categorical_cols and not df[categorical_cols].empty:
        cat_feats = enc.fit_transform(df[categorical_cols]).toarray()
    else:
        cat_feats = np.empty((len(df), 0))

    # Standard scaling for numeric features (if any)
    scaler = StandardScaler()
    if numeric_cols and not df[numeric_cols].empty:
        num_feats = scaler.fit_transform(df[numeric_cols])
    else:
        num_feats = np.empty((len(df), 0))

    # Save transformers for consistent preprocessing during inference
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(enc, os.path.join(save_dir, 'ohe_encoder.pkl'))
        joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
        print(f"Saved encoders to directory: {save_dir}")

    # Combine numeric and categorical features horizontally into one feature matrix
    # Ensure both are 2D
    # Ensure both are numpy arrays
    num_feats = np.array(num_feats)
    cat_feats = np.array(cat_feats)

    # Validate shapes
    if num_feats.ndim == 1:
        num_feats = num_feats.reshape(-1, 1)

    if cat_feats.ndim == 1:
        cat_feats = cat_feats.reshape(-1, 1)

    # Check if either is empty or invalid
    if num_feats.shape[0] == 0 or cat_feats.shape[0] == 0:
        raise ValueError(f"Empty input arrays! num_feats.shape={num_feats.shape}, cat_feats.shape={cat_feats.shape}")

    # Assert matching rows
    assert num_feats.shape[0] == cat_feats.shape[0], \
        f"Mismatch in rows: num_feats.shape={num_feats.shape}, cat_feats.shape={cat_feats.shape}"

    # Now safely stack
    features = np.column_stack([num_feats, cat_feats])

    # Extract target as numpy array
    target = df[target_col].values

    # Train-validation split, stratifying on target for balanced classes
    X_train, X_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target
    )

    print(f"Train set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
    print(f"Number of features after preprocessing: {features.shape[1]}")

    # Return as PyTorch tensors for immediate training use
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        features.shape[1]
    )


def load_transformers(encoder_dir):
    """
    Load previously saved OneHotEncoder and StandardScaler transformers.

    Args:
        encoder_dir (str): Directory where 'ohe_encoder.pkl' and 'scaler.pkl' are saved.

    Returns:
        enc (OneHotEncoder): Loaded encoder for categorical features.
        scaler (StandardScaler): Loaded scaler for numeric features.
    """
    enc = joblib.load(os.path.join(encoder_dir, 'ohe_encoder.pkl'))
    scaler = joblib.load(os.path.join(encoder_dir, 'scaler.pkl'))
    return enc, scaler
