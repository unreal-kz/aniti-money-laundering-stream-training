import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_json_files(selected_folders, json_dir, train_dir, test_dir, test_size=0.2):
    """
    Splits JSON files from selected folders into training and testing datasets.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    json_files = []
    for folder in selected_folders:
        folder_path = Path(json_dir) / str(folder)
        if folder_path.exists():
            json_files.extend(folder_path.glob("*.json"))

    if not json_files:
        raise ValueError("No JSON files found in the selected folders.")

    # Perform train-test split
    train_files, test_files = train_test_split(json_files, test_size=test_size, random_state=42)

    for file in train_files:
        os.rename(file, Path(train_dir) / file.name)
    for file in test_files:
        os.rename(file, Path(test_dir) / file.name)

    print(f"Training files: {len(train_files)}, Test files: {len(test_files)}")
