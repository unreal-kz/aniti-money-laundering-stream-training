import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_json_files(json_dir, train_dir, test_dir, test_size=0.2):
    """
    Splits JSON files into training and testing datasets.
    """
    # Clear train and test directories if they already exist
    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            for file in Path(dir_path).glob("*.json"):
                os.remove(file)
        else:
            os.makedirs(dir_path, exist_ok=True)

    # Perform train-test split
    json_files = list(Path(json_dir).glob("*.json"))
    train_files, test_files = train_test_split(json_files, test_size=test_size, random_state=42)

    for file in train_files:
        os.rename(file, Path(train_dir) / file.name)
    for file in test_files:
        os.rename(file, Path(test_dir) / file.name)

    print(f"Training files: {len(train_files)}, Test files: {len(test_files)}")
