import os
import subprocess
from scripts.split_train_test import split_json_files
from scripts.json_to_graph import json_to_graph_with_features
from scripts.train_model import train_xgboost_with_graph_features
from scripts.evaluate_model import evaluate_model
from pathlib import Path

def cleanup_directory(dir_path):
    """
    Clears all files in a directory.
    """
    if os.path.exists(dir_path):
        for file in Path(dir_path).glob("*"):
            os.remove(file)


def main():
    # Cleanup previous outputs if necessary
    # cleanup_directory("data/jsons/")
    cleanup_directory("data/train_jsons/")
    cleanup_directory("data/test_jsons/")
    cleanup_directory("data/graphs/")
    cleanup_directory("data/features/")
    # Paths and configurations
    # csv_file_path = "data/HI-Small_Trans.csv"  # Input raw CSV file
    csv_file_path = ""  # Input raw CSV file
    jsons_dir = "data/jsons/"            # Output directory for all JSON files
    train_json_dir = "data/train_jsons/" # Directory for training JSON files
    test_json_dir = "data/test_jsons/"   # Directory for test JSON files
    graphs_dir = "data/graphs/"          # Directory for storing graphs
    features_dir = "data/features/"      # Directory for graph features
    model_path = "models/xgboost_model.bst"  # Path to save the trained model
    account_json_mapping_dst = "data/account_json_mapping.json"  # Optional mapping
    chunksize = 10000  # Number of rows per chunk during CSV processing

    # XGBoost training parameters
    params = {
        "max_depth": 10,
        "eta": 0.01,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }
    rounds = 50

    # Step 0: Convert raw_data.csv to JSON (if not already done)
    if os.path.exists(csv_file_path):
        print(f"Converting {csv_file_path} to JSON files...")
        try:
            subprocess.run([
                "python", "scripts/csv_to_json.py",
                "--csv_file_path", csv_file_path,
                "--jsons_dir", jsons_dir,
                "--chunksize", str(chunksize),
                "--account_json_mapping", account_json_mapping_dst
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running csv_to_json.py: {e}")
            exit(1)
    else:
        print(f"CSV file not found: {csv_file_path}. Skipping CSV-to-JSON conversion.")
    
    selected_folders = [7]  # List of folder numbers to include for training
    # Step 1: Split JSON files into train and test datasets
    if not os.listdir(train_json_dir) or not os.listdir(test_json_dir):
        print(f"Splitting JSON files from folders {selected_folders} into training and testing sets...")
        split_json_files(selected_folders, jsons_dir, train_json_dir, test_json_dir, test_size=0.2)
    else:
        print("Train-test split already exists. Skipping this step.")

    # Step 2: Generate graphs and extract features from train JSONs
    print("Generating graphs and extracting features...")
    json_to_graph_with_features(train_json_dir, graphs_dir, features_dir)

    # Step 3: Train XGBoost model with extracted graph features
    print("Training XGBoost model with graph features...")
    train_xgboost_with_graph_features(features_dir, model_path, params, rounds)

    # Step 4: Evaluate the trained model on test JSON files
    print("Evaluating the model on test data...")
    evaluate_model(model_path, features_dir)  # Test features can also be extracted similarly


if __name__ == "__main__":
    main()
