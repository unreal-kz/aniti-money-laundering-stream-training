import os
import json
import pandas as pd
import numpy as np
import networkx as nx
from snapml import GraphFeaturePreprocessor
from pathlib import Path


def json_to_graph_with_features(jsons_dir, graphs_dir, features_dir):
    """
    Converts JSON files to graph structures, extracts features using SNAPML's GFP,
    and saves the features as CSV files.
    """
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    json_files = list(Path(jsons_dir).glob("*.json"))
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Create a graph from the JSON file
        G = nx.Graph()
        for transaction in data['transactions']:
            from_bank = transaction['From Bank']
            to_bank = transaction['To Bank']
            from_account = transaction['Account']
            to_account = transaction['Account.1']
            amount = transaction['Amount Received']
            timestamp = transaction['Timestamp']

            # Create unique account identifiers similar to the notebook
            source_account = f"{from_bank}_{from_account}"
            target_account = f"{to_bank}_{to_account}"

            # Convert timestamp to seconds since epoch
            timestamp_float = pd.to_datetime(timestamp, format='%Y/%m/%d %H:%M').value // 10**9

            # Add nodes and edge to the graph
            G.add_node(source_account, bank=from_bank, account=from_account)
            G.add_node(target_account, bank=to_bank, account=to_account)
            G.add_edge(source_account, target_account, weight=amount, timestamp=timestamp_float)


        columns_for_vertex_stats = [3, 7, 8, 9, 10, 11]
        # x_train_meta = x_train[non_float_columns]
        # x_train = x_train[correct_columns_order]
        # x_test_meta = x_test[non_float_columns]
        # x_test = x_test[correct_columns_order]

        #################################################
        hw = 5  # histogram size
        hw_bin_range = 5 # histogram bin size

        # Array used for specifying the bins of the pattern histogram
        histogram_range_init = [y * hw_bin_range + hw_bin_range + 1 for y in range(-1, hw)]
        print(histogram_range_init)
            
        vertex_stats_feats_types = [i for i in range(11)]

        # PAPER: not specified values --> vertex_stats_tw, time_window, max_no_edges
        params = {
            "num_threads": 12,            # number of software threads to be used (important for performance)
            "vertex_stats": True,         # produce vertex statistics

            "vertex_stats_tw": 12 * 60 * 60,        # time window to consider
            "time_window": 12 * 60 * 60,            # time window used if no pattern was specified
                                                    # transactions in largest time - time window
            
            "max_no_edges": 10,           # limit number of edges in detecting simple cycles
                                        # -1 means it is defined only using the time window
            
            "vertex_stats_cols": columns_for_vertex_stats,     
            # produce vertex statistics using the selected input columns
            # NOTE: Columns of the input numpy array used for generating vertex statistics features
            # NOTE: money amount, currency, etc. -- of source and target
            
            # features: 0:fan,1:deg,2:ratio,3:avg,4:sum,5:min,6:max,7:median,8:var,9:skew,10:kurtosis
            "vertex_stats_feats": vertex_stats_feats_types, # PAPER: computed based on Amount and Timestamp??
            
            # fan in/out parameters -- PAPER: NOT USED
            "fan": True,
            "fan_tw": 12 * 60 * 60, # seconds to hours (12 or 24)
            "fan_bins": histogram_range_init,
            
            # in/out degree parameters -- PAPER: NOT USED
            "degree": True,
            "degree_tw": 12 * 60 * 60, # seconds to hours (12 or 24)
            "degree_bins": histogram_range_init,
            
            # scatter gather parameters
            "scatter-gather": True,
            "scatter-gather_tw": 6 * 60 * 60, # seconds to hours
            "scatter-gather_bins": histogram_range_init,

            # # gather scatter parameters -- UNSUPPORTED
            # "gather-scatter": True,
            # "gather-scatter_tw": 6 * 60 * 60, # seconds to hours
            # "gather-scatter_bins": histogram_range_init,
            
            # temporal cycle parameters
            "temp-cycle": True,
            "temp-cycle_tw": 12 * 60 * 60, # seconds to hours (12 or 24)
            "temp-cycle_bins": histogram_range_init,
            
            # length-constrained simple cycle parameters
            "lc-cycle": True,
            "lc-cycle_tw": 12 * 60 * 60, # seconds to hours (12 or 24)
            "lc-cycle_bins": histogram_range_init,
            "lc-cycle_len": 10
        }


        # Convert graph to a contiguous NumPy array before fitting the GFP
        gfp = GraphFeaturePreprocessor()
        nodes = list(G.nodes)
        adjacency_matrix = nx.to_numpy_array(G, nodelist=nodes)
        adjacency_matrix_contiguous = np.ascontiguousarray(adjacency_matrix)
        print("Setting the parameters of the graph feature preprocessor ")
        gfp.set_params(params)
        # Fit the model on the contiguous array
        gfp.fit(adjacency_matrix_contiguous)

        # Extract features and convert to a DataFrame
        features = gfp.get_features()
        columns = [f"feature_{i}" for i in range(features.shape[1])]
        feature_df = pd.DataFrame(features, columns=columns)

        # Convert DataFrame to a contiguous NumPy array for XGBoost compatibility
        feature_array = np.ascontiguousarray(feature_df.values)

        # Save the graph
        graph_name = json_file.stem
        nx.write_gpickle(G, os.path.join(graphs_dir, f"{graph_name}_graph.gpickle"))

        # Save the features to a CSV
        feature_path = os.path.join(features_dir, f"{graph_name}_features.csv")
        feature_df.to_csv(feature_path, index=False)

        print(f"Graph and features saved for {graph_name}")

# Example usage
# json_to_graph_with_features('path/to/jsons', 'path/to/save/graphs', 'path/to/save/features')
