from snapml import GFP
import networkx as nx

# Create a simple graph
G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(2, 3)

# Fit the GFP and extract features
gfp = GFP()
gfp.fit(G)
features = gfp.get_features()
print(features)
