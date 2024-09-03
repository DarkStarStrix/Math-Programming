from typing import Any

import numpy as np
from numpy import ndarray, dtype
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle


def _get_routes(data, manager, routing, solution):
    routes = []
    for vehicle_id in range (data ['num_vehicles']):
        index = routing.Start (vehicle_id)
        route = []
        while not routing.IsEnd (index):
            node_index = manager.IndexToNode (index)
            route.append (data ['locations'] [node_index])
            index = solution.Value (routing.NextVar (index))
        routes.append (route)
    return routes


class DeliveryOptimizer:
    def __init__(self, num_vehicles, depot):
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.locations = []
        self.time_windows = []
        self.demands = []
        self.vehicle_capacities = []

    def add_location(self, location, time_window, demand):
        self.locations.append (location)
        self.time_windows.append (time_window)
        self.demands.append (demand)

    def set_vehicle_capacities(self, capacities):
        self.vehicle_capacities = capacities

    def _create_data_model(self):
        data = {}
        data ['locations'] = [self.depot] + self.locations
        data ['num_vehicles'] = self.num_vehicles
        data ['depot'] = 0
        data ['time_windows'] = [(0, 0)] + self.time_windows
        data ['demands'] = [0] + self.demands
        data ['vehicle_capacities'] = self.vehicle_capacities

        # Calculate distance matrix
        num_locations = len (data ['locations'])
        data ['distance_matrix'] = np.zeros ((num_locations, num_locations))
        for i in range (num_locations):
            for j in range (num_locations):
                data ['distance_matrix'] [i] [j] = np.linalg.norm (
                    np.array (data ['locations'] [i]) - np.array (data ['locations'] [j]))

        return data

    def optimize_routes(self):
        data = self._create_data_model ()
        manager = pywrapcp.RoutingIndexManager (
            len (data ['distance_matrix']), data ['num_vehicles'], data ['depot'])
        routing = pywrapcp.RoutingModel (manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode (from_index)
            to_node = manager.IndexToNode (to_index)
            return data ['distance_matrix'] [from_node] [to_node]

        transit_callback_index = routing.RegisterTransitCallback (distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles (transit_callback_index)

        # Add capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode (from_index)
            return data ['demands'] [from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback (demand_callback)
        routing.AddDimensionWithVehicleCapacity (
            demand_callback_index,
            0,  # null capacity slack
            data ['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Add time window constraints
        time = 'Time'
        routing.AddDimension (
            transit_callback_index,
            30,  # allow waiting time
            300,  # maximum time per vehicle
            False,  # don't force start cumul to zero
            time)
        time_dimension = routing.GetDimensionOrDie (time)

        for location_idx, time_window in enumerate (data ['time_windows']):
            if location_idx == 0:
                continue
            index = manager.NodeToIndex (location_idx)
            time_dimension.CumulVar (index).SetRange (time_window [0], time_window [1])

        # Set first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters ()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem
        solution = routing.SolveWithParameters (search_parameters)

        if solution:
            return _get_routes (data, manager, routing, solution)
        else:
            return None

    def visualize_routes(self, routes):
        plt.figure (figsize=(15, 12))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

        # Create custom colormap for demand heatmap
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list ("custom", ["#FFF3E0", "#FF5722"], N=n_bins)

        # Plot heatmap for demand intensity
        x = np.array ([loc [0] for loc in self.locations])
        y = np.array ([loc [1] for loc in self.locations])
        z = np.array (self.demands)
        plt.hist2d (x, y, weights=z, bins=20, cmap=cmap, alpha=0.3)
        plt.colorbar (label='Demand Intensity')

        # Plot depot
        plt.plot (self.depot [0], self.depot [1], 'ko', markersize=15, label='Depot')

        # Plot locations
        locations = np.array (self.locations)
        scatter = plt.scatter (locations [:, 0], locations [:, 1], c=self.demands,
                               cmap='viridis', s=100, edgecolor='black', label='Delivery Locations')
        plt.colorbar (scatter, label='Demand')

        # Plot routes
        for i, route in enumerate (routes):
            route_arr = np.array (route)
            plt.plot (route_arr [:, 0], route_arr [:, 1], colors [i % len (colors)] + '-o',
                      linewidth=2, markersize=8, label=f'Vehicle {i + 1}')

            # Add direction arrows
            for j in range (len (route) - 1):
                plt.annotate ('', xy=route [j + 1], xytext=route [j],
                              arrowprops=dict (arrowstyle='->', color=colors [i % len (colors)], lw=2),
                              va='center', ha='center')

        # Add labels for locations
        for i, loc in enumerate (self.locations):
            plt.annotate (f'L{i + 1}', xy=(loc [0], loc [1]), xytext=(3, 3), textcoords='offset points')

        # Add a circle to represent service area
        for loc, demand in zip (self.locations, self.demands):
            circle = Circle ((loc [0], loc [1]), demand / 2, fill=False, color='r', linestyle='dashed')
            plt.gca ().add_artist (circle)

        plt.title ('Optimized Delivery Routes with Demand Heatmap')
        plt.xlabel ('X coordinate')
        plt.ylabel ('Y coordinate')
        plt.legend (loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid (True)
        plt.tight_layout ()
        plt.show ()


# Example usage with more locations
optimizer = DeliveryOptimizer (num_vehicles=4, depot=(0, 0))
optimizer.add_location ((1, 5), (0, 100), 2)
optimizer.add_location ((2, 3), (0, 100), 3)
optimizer.add_location ((4, 1), (0, 100), 1)
optimizer.add_location ((3, 6), (0, 100), 4)
optimizer.add_location ((5, 2), (0, 100), 2)
optimizer.add_location ((-2, 4), (0, 100), 3)
optimizer.add_location ((-1, -3), (0, 100), 1)
optimizer.add_location ((3, -2), (0, 100), 2)
optimizer.add_location ((-4, 2), (0, 100), 4)
optimizer.add_location ((2, -4), (0, 100), 3)
optimizer.set_vehicle_capacities ([10, 10, 10, 10])

optimized_routes = optimizer.optimize_routes ()
print ("Optimized routes:", optimized_routes)

# Visualize the routes
optimizer.visualize_routes (optimized_routes)
