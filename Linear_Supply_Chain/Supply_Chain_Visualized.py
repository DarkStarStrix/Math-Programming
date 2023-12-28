# Create a directed graph based on the parsed transportation results
import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Adding nodes for warehouses and stores
warehouses = ['Warehouse1', 'Warehouse2']
stores = ['Store1', 'Store2']
for w in warehouses:
    G.add_node(w, pos=(1, warehouses.index(w) * 2), node_color='skyblue')
for s in stores:
    G.add_node(s, pos=(2, stores.index(s) * 2), node_color='lightgreen')

# Adding edges with transportation quantities
for (w, s, p), quantity in transport_results.items():
    if quantity > 0:
        G.add_edge(w, s, weight=quantity, label=f"{p}: {quantity}")

# Positioning and coloring
pos = nx.get_node_attributes(G, 'pos')
node_color = [G.nodes[n]['node_color'] for n in G.nodes]

# Drawing the network
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=3000,
        font_size=12, font_weight='bold', edge_color='gray', width=1.5)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Show plot
plt.title('Supply Chain Transportation Network')
plt.axis('off')
plt.show()
