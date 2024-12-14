import os
import json
from torch_geometric.data import HeteroData
import torch
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

# Load BERT model and tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

def embed_text(texts):
    """Generate BERT embeddings for a list of texts."""
    with torch.no_grad():
        encoded_input = bert_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        output = bert_model(**encoded_input)
        # Use the [CLS] token representation
        embeddings = output.last_hidden_state[:, 0, :]  # Shape: [num_texts, hidden_size]
    return embeddings

# Step 1: Parse the PHEME File Structure
def parse_pheme(base_path):
    data = []  # List to hold all graph data
    for event in os.listdir(base_path):
        event_path = os.path.join(base_path, event)
        if not os.path.isdir(event_path):
            continue  # Skip non-directory files like .DS_Store
        for label_type in ['rumours', 'non-rumours']:
            label = 0 if label_type == 'rumours' else 1  # Encode rumor: 0, non-rumor: 1
            label_path = os.path.join(event_path, label_type)
            if not os.path.isdir(label_path):
                continue  # Skip if not a directory
            for graph_id in os.listdir(label_path):
                graph_path = os.path.join(label_path, graph_id)
                if not os.path.isdir(graph_path):
                    continue  # Skip non-directory files
                annotation_file = os.path.join(graph_path, "annotation.json")
                source_tweet_path = os.path.join(graph_path, "source-tweets")
                reactions_path = os.path.join(graph_path, "reactions")
                structure_file = os.path.join(graph_path, "structure.json")

                data.append({
                    "graph_id": graph_id,
                    "event": event,
                    "label": label,
                    "annotation": annotation_file,
                    "source_tweet": source_tweet_path,
                    "reactions": reactions_path,
                    "structure": structure_file,
                })
    return data

# Step 2: Load Tweet Data
def load_tweet_data(tweet_file):
    with open(tweet_file, 'r') as f:
        tweet_data = json.load(f)
    return {
        "text": tweet_data["text"],
        "created_at": tweet_data["created_at"],
        "id": tweet_data["id"],
        "favorite_count": tweet_data.get("favorite_count", 0),
        "retweet_count": tweet_data.get("retweet_count", 0),
        "user_id": tweet_data["user"]["id"],
        "user_followers_count": tweet_data["user"].get("followers_count", 0),
        "user_friends_count": tweet_data["user"].get("friends_count", 0)
    }

# Step 3: Parse the Structure File
def parse_structure(structure_file):
    with open(structure_file, 'r') as f:
        structure = json.load(f)
    edges = []
    for parent, children in structure.items():
        for child in children:
            edges.append((parent, child))
    return edges

# Step 4: Build Heterogeneous Graph
def build_hetero_graph(graph_data):
    # Parse the source tweet
    source_tweet_file = os.path.join(graph_data["source_tweet"], os.listdir(graph_data["source_tweet"])[0])
    source_data = load_tweet_data(source_tweet_file)

    # Parse the reactions
    reactions = []
    for reaction_file in os.listdir(graph_data["reactions"]):
        reaction_path = os.path.join(graph_data["reactions"], reaction_file)
        reactions.append(load_tweet_data(reaction_path))

    # Extract tweet texts
    tweets = [source_data] + reactions
    tweet_texts = [tweet["text"] for tweet in tweets]

    # Embed tweet texts using BERT
    text_embeddings = embed_text(tweet_texts)  # Shape: [num_tweets, hidden_size]

    # Build the HeteroData object
    hetero_graph = HeteroData()

    # Add tweet nodes
    hetero_graph["tweet"].x = torch.cat([
        torch.tensor([[tweet["favorite_count"], tweet["retweet_count"]]], dtype=torch.float) for tweet in tweets
    ], dim=0)
    hetero_graph["tweet"].text_embeddings = text_embeddings  # Add text embeddings as features

    # Add user nodes
    user_ids = list({tweet["user_id"] for tweet in tweets})
    hetero_graph["user"].x = torch.tensor([[0] for _ in user_ids], dtype=torch.float)  # Example user features

    # Map user IDs to indices
    user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}

    # Map tweet IDs to indices
    tweet_id_map = {str(tweet["id"]): idx for idx, tweet in enumerate(tweets)}

    # Add tweet-to-tweet edges
    filtered_edges = [
        (tweet_id_map[src], tweet_id_map[dst]) for src, dst in parse_structure(graph_data["structure"])
        if src in tweet_id_map and dst in tweet_id_map
    ]
    hetero_graph[("tweet", "replies_to", "tweet")].edge_index = torch.tensor(filtered_edges, dtype=torch.long).t()

    # Add user-to-tweet edges
    hetero_graph[("user", "authors", "tweet")].edge_index = torch.tensor([
        [user_id_map[tweet["user_id"]], idx] for idx, tweet in enumerate(tweets)
    ], dtype=torch.long).t()

    # Add label
    hetero_graph["graph_label"] = torch.tensor([graph_data["label"]], dtype=torch.long)

    return hetero_graph

# Visualization Function
def visualize_graph(hetero_graph, title="Graph Visualization"):
    nx_graph = nx.DiGraph()

    # Add tweet nodes
    tweet_offset = 0
    for i in range(hetero_graph["tweet"].x.size(0)):
        nx_graph.add_node(i + tweet_offset, node_type="tweet")

    # Add user nodes
    user_offset = hetero_graph["tweet"].x.size(0)
    for i in range(hetero_graph["user"].x.size(0)):
        nx_graph.add_node(i + user_offset, node_type="user")

    # Add tweet-to-tweet edges
    for src, dst in hetero_graph[("tweet", "replies_to", "tweet")].edge_index.t().tolist():
        nx_graph.add_edge(src + tweet_offset, dst + tweet_offset, edge_type="replies_to")

    # Add user-to-tweet edges
    for src, dst in hetero_graph[("user", "authors", "tweet")].edge_index.t().tolist():
        nx_graph.add_edge(src + user_offset, dst + tweet_offset, edge_type="authors")

    # Visualize
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(nx_graph)  # Layout for visualization

    # Draw nodes with different colors
    node_colors = ["blue" if nx_graph.nodes[n]["node_type"] == "tweet" else "green" for n in nx_graph.nodes]
    nx.draw(nx_graph, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(nx_graph, "edge_type")
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color="red")

    plt.title(title)
    plt.show()

# Step 5: Dataset Class
def build_dataset(base_path, save_dir=None):
    graph_data = parse_pheme(base_path)
    event_graphs = {}

    for graph in graph_data:
        event = graph["event"]
        hetero_graph = build_hetero_graph(graph)
        if event not in event_graphs:
            event_graphs[event] = []
        event_graphs[event].append(hetero_graph)

    # Save each event's graphs to a separate pickle file
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for event, graphs in event_graphs.items():
            save_path = os.path.join(save_dir, f"{event}_graphs.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Saved {len(graphs)} graphs for event '{event}' to {save_path}")

    return event_graphs

# Example of running all functions
def main():
    dataset_path = "/Users/andrewxu/Downloads/PHEME-dataset/data/all-rnr-annotated-threads"  # Replace with your dataset path
    save_dir = "pheme_graphs"  # Directory to save individual event graphs
    graphs_by_event = build_dataset(dataset_path, save_dir)

    # Example usage: Print the first graph for each event and visualize
    for event, graphs in graphs_by_event.items():
        print(f"Event: {event}, Number of Graphs: {len(graphs)}")
        print("First Graph:")
        print(graphs[0])
        visualize_graph(graphs[0], title=f"Graph Visualization for {event}")

if __name__ == "__main__":
    main()
