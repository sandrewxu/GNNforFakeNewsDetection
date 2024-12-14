import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch

def parse_pheme_data(base_dir, output_dir, embedding_dim=768, edge_feature_dim=16, device="cpu"):
    edges = []
    edge_idx = 1
    node_features = {}
    node_idx_map = {}
    graph_label = {}

    # Iterate through events
    for event in os.listdir(base_dir):
        event_path = os.path.join(base_dir, event)
        if not os.path.isdir(event_path):  # Skip non-directory files
            continue

        for label_type in ['rumours', 'non-rumours']:
            label = 1 if label_type == 'rumours' else 0
            label_path = os.path.join(event_path, label_type)
            if not os.path.isdir(label_path):  # Skip non-directory files
                continue

            for graph_id in os.listdir(label_path):
                graph_path = os.path.join(label_path, graph_id)
                if not os.path.isdir(graph_path):  # Skip non-directory files
                    continue

                # Set graph label
                graph_label[graph_id] = label

                # Parse source tweet
                source_path = os.path.join(graph_path, 'source-tweets')
                source_file = next((f for f in os.listdir(source_path) if f.endswith('.json') and not f.startswith('._')), None)
                if source_file:
                    try:
                        with open(os.path.join(source_path, source_file), 'r', encoding='utf-8') as f:
                            source_data = json.load(f)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        print(f"Skipping invalid or non-UTF-8 file: {source_file}")
                        continue

                    source_id = source_data['id']
                    timestamp = int(datetime.strptime(source_data['created_at'], "%a %b %d %H:%M:%S %z %Y").timestamp())

                    # Map source ID to node index
                    if source_id not in node_idx_map:
                        node_idx_map[source_id] = len(node_idx_map) + 1
                        embedding = np.random.randn(embedding_dim)  # Use random embeddings for now
                        node_features[node_idx_map[source_id]] = embedding

                # Parse reactions
                reactions_path = os.path.join(graph_path, 'reactions')
                for reaction_file in os.listdir(reactions_path):
                    if not reaction_file.endswith('.json') or reaction_file.startswith('._'):
                        continue
                    try:
                        with open(os.path.join(reactions_path, reaction_file), 'r', encoding='utf-8') as f:
                            reaction_data = json.load(f)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        print(f"Skipping invalid or non-UTF-8 file: {reaction_file}")
                        continue

                    reaction_id = reaction_data['id']
                    reaction_time = int(datetime.strptime(reaction_data['created_at'], "%a %b %d %H:%M:%S %z %Y").timestamp())

                    # Map reaction ID to node index
                    if reaction_id not in node_idx_map:
                        node_idx_map[reaction_id] = len(node_idx_map) + 1
                        embedding = np.random.randn(embedding_dim)  # Use random embeddings for now
                        node_features[node_idx_map[reaction_id]] = embedding

                    # Add edge
                    edges.append({
                        'u': node_idx_map[source_id],
                        'i': node_idx_map[reaction_id],
                        'ts': reaction_time,
                        'label': label,
                        'idx': edge_idx
                    })
                    edge_idx += 1

    # Save edges to CSV
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(os.path.join(output_dir, 'ml_pheme.csv'), index=False)

    # Save node features to NumPy
    max_node_idx = max(node_features.keys())
    node_feature_matrix = np.zeros((max_node_idx + 1, embedding_dim))
    for node_id, feature in node_features.items():
        node_feature_matrix[node_id] = feature
    np.save(os.path.join(output_dir, 'ml_pheme_node.npy'), node_feature_matrix)

    # Save edge features to NumPy
    edge_feature_matrix = np.random.randn(len(edges) + 1, edge_feature_dim)
    np.save(os.path.join(output_dir, 'ml_pheme.npy'), edge_feature_matrix)

    print(f"Processed data saved in {output_dir}")
    
# Paths
base_dir = "/Users/andrewxu/Downloads/PHEME-dataset/data/all-rnr-annotated-threads"
output_dir = "processed"
os.makedirs(output_dir, exist_ok=True)

# Process data
parse_pheme_data(base_dir, output_dir, device="cuda" if torch.cuda.is_available() else "cpu")
