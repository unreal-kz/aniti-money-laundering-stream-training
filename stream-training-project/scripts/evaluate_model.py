import xgboost as xgb
import pandas as pd
from pathlib import Path
from scripts.json_to_graph import json_to_graph_with_features


def evaluate_model(model_path, test_features_dir, test_json_dir=None, graphs_dir=None):
    """
    Loads a saved model and evaluates it on the test dataset.
    If test features are not precomputed, extract them first.
    """
    if test_json_dir and graphs_dir:
        print("Extracting features from test JSON files...")
        json_to_graph_with_features(test_json_dir, graphs_dir, test_features_dir)

    model = xgb.Booster()
    model.load_model(model_path)

    test_files = list(Path(test_features_dir).glob("*.csv"))
    all_test_features = pd.concat([pd.read_csv(f) for f in test_files], axis=0)

    X_test = all_test_features.drop(columns=['Is Laundering'])
    y_test = all_test_features['Is Laundering']

    dtest = xgb.DMatrix(X_test, label=y_test)
    preds = model.predict(dtest)

    print(f"Predictions: {preds}")
