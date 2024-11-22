import os
import argparse
import pandas as pd
import json
from collections import defaultdict
from uuid import uuid4


def process_chunk(sub_df, jsons_dir, idx_folder, account_json_mapping):
    os.makedirs(os.path.join(jsons_dir, idx_folder), exist_ok=True)
    for _, row in sub_df.iterrows():
        account_a, account_b = row['Account'], row['Account.1']
        if account_a in account_json_mapping or account_b in account_json_mapping:
            if account_a in account_json_mapping:
                account_a_path = os.path.join(jsons_dir, account_json_mapping[account_a]['folder'],
                                              account_json_mapping[account_a]['uuid'] + '.json')
                with open(account_a_path, 'r') as f:
                    data = json.load(f)
                data['transactions'].append(row.to_dict())
                with open(account_a_path, 'w') as f:
                    json.dump(data, f, indent=4)
            if account_b in account_json_mapping:
                account_b_path = os.path.join(jsons_dir, account_json_mapping[account_b]['folder'],
                                              account_json_mapping[account_b]['uuid'] + '.json')
                with open(account_b_path, 'r') as f:
                    data = json.load(f)
                data['transactions'].append(row.to_dict())
                with open(account_b_path, 'w') as f:
                    json.dump(data, f, indent=4)
        else:
            uuid = str(uuid4())
            account_json_mapping[account_a] = {'uuid': uuid, 'folder': idx_folder}
            account_json_mapping[account_b] = {'uuid': uuid, 'folder': idx_folder}
            with open(os.path.join(jsons_dir, idx_folder, uuid + '.json'), 'w') as f:
                json.dump({'transactions': [row.to_dict()]}, f, indent=4)


def main(csv_file_path, jsons_dir, chunksize, account_json_mapping_dst):
    os.makedirs(jsons_dir, exist_ok=True)
    account_json_mapping = defaultdict(list)
    with pd.read_csv(csv_file_path, chunksize=chunksize) as reader:
        total_chunks = sum(1 for _ in reader)
    
    with pd.read_csv(csv_file_path, chunksize=chunksize) as reader:
        idx_folder = 0
        for chunk in reader:
            idx_folder += 1
            process_chunk(chunk, jsons_dir, str(idx_folder), account_json_mapping)
    with open(account_json_mapping_dst, 'w') as f:
        json.dump(account_json_mapping, f, indent=4)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--csv_file_path', type=str, required=True, help='Path to the input CSV file')
    args.add_argument('--jsons_dir', type=str, default='jsons', help='Directory to save JSON files')
    args.add_argument('--chunksize', type=int, default=10000, help='Number of rows per chunk to process')
    args.add_argument('--account_json_mapping', type=str, default='account_json_mapping.json',
                      help='Path to save account_json_mapping')
    parsed_args = args.parse_args()
    main(parsed_args.csv_file_path, parsed_args.jsons_dir, parsed_args.chunksize, parsed_args.account_json_mapping)
