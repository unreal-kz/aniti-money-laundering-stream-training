import os
import pandas as pd
import xgboost as xgb
from pathlib import Path


def train_xgboost_with_graph_features(features_dir, model_path, params, num_rounds):
    """
    Trains an XGBoost model using features extracted from graph structures.
    """
    # Combine all feature files
    feature_files = list(Path(features_dir).glob("*.csv"))
    if not feature_files:
        raise ValueError("No feature files found in the specified directory.")

    all_features = pd.concat([pd.read_csv(f) for f in feature_files], ignore_index=True)

    # Extract labels (for simplicity, generating random labels for now)
    # Replace this with your actual labels.
    labels = pd.Series([0 if i % 2 == 0 else 1 for i in range(len(all_features))])

    # Convert to XGBoost DMatrix
    dtrain = xgb.DMatrix(all_features, label=labels)

    # Train the XGBoost model
    model = xgb.train(params, dtrain, num_boost_round=num_rounds)

    # Save the model
    model.save_model(model_path)
    print(f"Model trained and saved at {model_path}")
