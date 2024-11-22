import xgboost as xgb
import pandas as pd
from pathlib import Path


def train_xgboost_with_graph_features(features_dir, model_path, params, rounds):
    feature_files = list(Path(features_dir).glob("*.csv"))
    all_features = pd.concat([pd.read_csv(f) for f in feature_files], axis=0)

    X = all_features.drop(columns=['Is Laundering'])
    y = all_features['Is Laundering']

    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=rounds)

    model.save_model(model_path)
    print(f"Model saved at {model_path}")
