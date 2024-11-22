import json
import pandas as pd
from pathlib import Path

def stream_json_files(folder_path, batch_size):
    """
    Streams JSON files in batches for memory-efficient loading.
    """
    files = Path(folder_path).glob("*.json")
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            transactions = pd.DataFrame(data['transactions'])
            for i in range(0, len(transactions), batch_size):
                yield transactions.iloc[i:i + batch_size]