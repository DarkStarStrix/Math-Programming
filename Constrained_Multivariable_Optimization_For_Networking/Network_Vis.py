import matplotlib.pyplot as plt
from Networking import model, locations

# Assuming 'locations' has corresponding x, y coordinates for plotting
location_coords = {'Loc1': (1, 1), 'Loc2': (2, 3), 'Loc3': (3, 2)}
coverage_areas = {'Loc1': 100, 'Loc2': 200, 'Loc3': 150}

# Gather the node build decisions and capacities
node_build_decisions = {loc: model.build_node[loc].value for loc in locations}
node_capacities = {loc: model.node_capacity[loc].value for loc in locations}

# Define colors and markers for each location
colors = {'Loc1': 'red', 'Loc2': 'blue', 'Loc3': 'green'}
markers = {'Loc1': 'o', 'Loc2': 's', 'Loc3': '^'}

# Plotting the network design
plt.figure(figsize=(12, 8))
for loc in locations:
    if node_build_decisions[loc] > 0:
        coverage = coverage_areas[loc]
        capacity = node_capacities[loc]
        x, y = location_coords[loc]
        plt.scatter(x, y, s=coverage, alpha=0.5, c=colors[loc], marker=markers[loc], label=f"{loc} (Capacity: {capacity})")

plt.title('Network Design Visualization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()
