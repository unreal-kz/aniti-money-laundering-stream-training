from snapml import GraphFeaturePreprocessor as GFP
import networkx as nx

# Create a small test graph
G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(2, 3)

# Try initializing the GFP class and fitting the graph
try:
    gfp = GFP()
    gfp.fit(G)
    features = gfp.get_features()
    print("Graph features:", features)
except Exception as e:
    print("SNAPML Error:", e)
