import pandas as pd

# Load the cleaned dataset
cleaned_data = pd.read_csv("cleaned_data.csv")

# Export nodes file (with predictions)
nodes = cleaned_data.copy()
nodes.to_csv("nodes.csv", index=False)

# Prepare edges file (example: fully connected graph)
edges = []

# Loop through all possible connections between nodes
for i, source_node in enumerate(nodes.index):
    for j, target_node in enumerate(nodes.index):
        if i != j:  # Avoid self-loops
            edges.append({'source': source_node, 'target': target_node})

# Convert edges list to DataFrame
edges_df = pd.DataFrame(edges)

# Export edges file
edges_df.to_csv("edges.csv", index=False)
