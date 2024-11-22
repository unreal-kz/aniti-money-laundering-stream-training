import os
import json
from pathlib import Path
from snapml import GraphFeaturePreprocessor as GFP
import networkx as nx
import pandas as pd


def json_to_graph_with_features(jsons_dir, graphs_dir, features_dir):
    """
    Converts JSON files to graph structures and extracts features using SNAPML's GFP.
    """
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    json_files = list(Path(jsons_dir).glob("*.json"))
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Validate transactions (remove duplicates)
        unique_transactions = {json.dumps(txn, sort_keys=True) for txn in data['transactions']}
        transactions = [json.loads(txn) for txn in unique_transactions]

        # Create a graph
        G = nx.Graph()
        for transaction in transactions:
            from_account = transaction['From Bank']
            to_account = transaction['To Bank']
            amount = transaction['Amount Received']
            G.add_edge(from_account, to_account, weight=amount)

        # Apply SNAPML's GFP
        gfp = GFP()
        gfp.fit(G)

        # Save graph and features
        graph_name = json_file.stem
        nx.write_gpickle(G, os.path.join(graphs_dir, f"{graph_name}_graph.gpickle"))

        features = gfp.get_features()
        feature_df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
        feature_path = os.path.join(features_dir, f"{graph_name}_features.csv")
        feature_df.to_csv(feature_path, index=False)

        print(f"Graph and features saved for {graph_name}")
