# Explanation: of the Code written in the codebase of the project

## Outlining the goals of the project
This project aims to create a unified resource for all who are interested in operations research and Each topic contains a brief overview, a list of resources, and a list of related topics. The project is open source and contributions are welcome.

## Linear Programming Model in GAMS
GAMS (General Algebraic Modeling System) and defines a linear programming model. The goal is to determine the optimal number of comedy and football ads to purchase to minimize the objective function value, subject to given constraints.

First, the code declares three variables: `x1`, `x2`, and `z`. `x1` represents the number of comedy ads purchased, `x2` represents the number of football ads purchased, and `z` is the objective function value.

```gams
Variable
     x1   Number of comedy ads purchased
     x2   Number of football ads purchased
     z    Objective function value
;
```

Next, the objective function is defined using an equation named `obj`. The objective function aims to minimize the total cost, which is calculated as `50*x1 + 100*x2`.

```gams
Equation     obj    Objective function;
             obj..  z =e= 50*x1 + 100*x2;
```

The code then defines two constraints, `eq1` and `eq2`. The first constraint ensures that the total exposure from comedy and football ads is at least 28, represented by the equation `7*x1 + 2*x2 =g= 28`. The second constraint ensures that the total exposure from comedy and football ads is at least 24, represented by the equation `2*x1 + 12*x2 =g= 24`.

```gams
Equation     eq1    Constraint 1;
             eq1..  7*x1 + 2*x2 =g= 28;
             
Equation     eq2    Constraint 2;
             eq2..  2*x1 + 12*x2 =g= 24;
```

The model is then assembled using the `model` statement, which includes all the defined equations.

```gams
model example_1 /all/;
```

Finally, the code specifies the solver to be used (`cplex`) and solves the model using linear programming to minimize the objective function `z`.

```gams
option lp = cplex;
solve example_1 using lp minimization z;
```

In summary, this GAMS code sets up a linear programming model to minimize the cost of purchasing ads while satisfying the given constraints on exposure.

## Linear Programming Model 2 in GAMS
This code is written in GAMS (General Algebraic Modeling System) and defines a linear programming model to determine the optimal number of workers starting on each day of the week to minimize the total number of workers, while satisfying daily requirements.

First, the code declares eight variables: `x1` through `x7` represent the number of workers beginning work on each day from Monday to Sunday, respectively. The variable `z` represents the objective function value, which is the total number of workers.

```gams
Variable
      x1 Number of worker begninning work on Monday
      x2 Number of worker begninning work on Tuesday
      x3 Number of worker begninning work on Wednesday
      x4 Number of worker begninning work on Thursday
      x5 Number of worker begninning work on Friday
      x6 Number of worker begninning work on Saturday
      x7 Number of worker begninning work on Sunday
      
      z Objective fnction value
```

Next, the objective function is defined using an equation named `obj`. The objective function aims to minimize the total number of workers, calculated as the sum of `x1` through `x7`.

```gams
Equation    obj     Objective function;
            obj..   z =e= x1 + x2 + x3 + x4 + x5 + x6 + x7;
```

The code then defines seven constraints, `eq1` through `eq7`, each representing the minimum number of workers required on each day of the week. For example, the constraint for Monday ensures that the sum of workers starting on Monday, Thursday, Friday, Saturday, and Sunday is at least 17.

```gams
Equation    eq1     Monday requiremnet;
            eq1..   x1 + x4 + x5 + x6 + x7 =g= 17;
```

Similar constraints are defined for the other days of the week, ensuring that the required number of workers is met for each day.

```gams
Equation    eq2     Tuesday requiremnet;
            eq2..   x1 + x2 + x5 + x6 + x7 =g= 13;
```

Additionally, the code includes non-negativity constraints for each variable to ensure that the number of workers starting on any given day is not negative.

```gams
Equation    eq8        Non Monday;
            eq8..      x1 =g= 0;
```

The model is then assembled using the `model` statement, which includes all the defined equations.

```gams
model example_2 /all/;
```

Finally, the code specifies the solver to be used (`cplex`) and solves the model using linear programming to minimize the objective function `z`.

```gams
option lp =cplex;
solve example_2 using lp minimization z;
```

In summary, this GAMS code sets up a linear programming model to minimize the total number of workers while satisfying the daily requirements for each day of the week.

## Linear Programming Model 3 in GAMS
This code is written in GAMS (General Algebraic Modeling System) and defines a linear programming model to determine the optimal number of ads to purchase for two types of ads: comedy and football. The objective is to minimize the total cost while meeting the minimum viewership requirements from two customer bases: HIW and HIM.

First, the code declares two sets: `c` for customer bases and `a` for ad types. The customer bases are `HIW` and `HIM`, and the ad types are `comedy` and `football`.

```gams
Sets
   c  /HIW
       HIm/
   a  /comedy
       football/;
```

Next, the code declares two variables: `x(a)` represents the number of ads of type `a` to be purchased, and `z` is the objective function value, which represents the total cost.

```gams
Variables
        x(a)    Number of ads of type a to be purcahsed
        z       Objective Function value;
```

The code then declares a parameter `theta(a)` to represent the cost of running each type of ad. The cost for a comedy ad is 50, and for a football ad, it is 100.

```gams
Parameter
        theta(a)     Cost of Running ad of type a
        /comedy 50
         football    100/;
```

A table `mu(a,c)` is defined to represent the number of viewers for each ad type from each customer base. For example, a comedy ad reaches 7 viewers from HIW and 2 from HIM, while a football ad reaches 2 viewers from HIW and 12 from HIM.

```gams
Table
        mu(a,c)      Number of viewers for ad of type a from customer base c
         
              HIW   HIM
comedy        7     2
football      2     12;
```

Another parameter `alpha(c)` is declared to represent the minimum viewership required from each customer base. HIW requires a minimum of 28 viewers, and HIM requires 24.

```gams
Parameter
        alpha(c)    Minimum viewership from customer base c
        /HIW    28
        HIm     24/;
```

The objective function is defined using an equation named `obj`. The objective function aims to minimize the total cost, calculated as the sum of the cost of each ad type multiplied by the number of ads purchased.

```gams
Equation    obj     Objective function;
            obj..   z =e= sum(a,theta(a)*x(a));
```

The code then defines a constraint `eq1(c)` to ensure that the total viewership from the purchased ads meets the minimum requirements for each customer base. This is done by summing the product of the number of ads and the number of viewers for each ad type and ensuring it is greater than or equal to the minimum viewership.

```gams
Equation    eq1(c)  Constrait 1;
            eq1(c)..  sum(a,mu(a,c)*x(a)) =g= alpha(c);
```

The model is assembled using the `model` statement, which includes all the defined equations.

```gams
model example_3 /all/;
```

Finally, the code specifies the solver to be used (`cplex`) and solves the model using linear programming to minimize the objective function `z`.

```gams
option lp = cplex;
solve example_3 using lp minimization z;
```

In summary, this GAMS code sets up a linear programming model to minimize the cost of purchasing ads while satisfying the minimum viewership requirements for two customer bases.

## Networking Optimization Model 
The selected code is a Python script that visualizes a network design using the `matplotlib` library. It imports necessary modules and data from a module named `Networking`, which includes a model and locations.

First, the script defines the coordinates and coverage areas for each location. The `location_coords` dictionary maps each location to its respective (x, y) coordinates, while the `coverage_areas` dictionary specifies the coverage area for each location.

```python
location_coords = {'Loc1': (1, 1), 'Loc2': (2, 3), 'Loc3': (3, 2)}
coverage_areas = {'Loc1': 100, 'Loc2': 200, 'Loc3': 150}
```

Next, the script gathers the node build decisions and capacities from the model. The `node_build_decisions` dictionary stores whether a node is built at each location, and the `node_capacities` dictionary stores the capacity of each node.

```python
node_build_decisions = {loc: model.build_node[loc].value for loc in locations}
node_capacities = {loc: model.node_capacity[loc].value for loc in locations}
```

The script then defines colors and markers for each location to differentiate them visually on the plot.

```python
colors = {'Loc1': 'red', 'Loc2': 'blue', 'Loc3': 'green'}
markers = {'Loc1': 'o', 'Loc2': 's', 'Loc3': '^'}
```

The main part of the script involves plotting the network design. It creates a figure with a specified size and iterates over each location. If a node is built at a location, it plots the location on the scatter plot with its corresponding coverage area, color, and marker. The label for each point includes the location name and its capacity.

```python
plt.figure(figsize=(12, 8))
for loc in locations:
    if node_build_decisions[loc] > 0:
        coverage = coverage_areas[loc]
        capacity = node_capacities[loc]
        x, y = location_coords[loc]
        plt.scatter(x, y, s=coverage, alpha=0.5, c=colors[loc], marker=markers[loc], label=f"{loc} (Capacity: {capacity})")
```

Finally, the script adds a title, labels for the x and y axes, a legend, and a grid to the plot. It then displays the plot using `plt.show()`.

```python
plt.title('Network Design Visualization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()
```

In summary, this code visualizes the network design by plotting the locations, their coverage areas, and capacities on a scatter plot, making it easier to understand the network's structure and node distribution.

## Dynamic Programming Model
The selected code is a Python script that implements a vehicle routing problem (VRP) solver using Google's OR-Tools and visualizes the optimized routes using `matplotlib`. The script is structured into a class named `DeliveryOptimizer` and several helper functions.

First, the necessary modules are imported, including `numpy` for numerical operations, `ortools.constraint_solver` for the VRP solver, and `matplotlib` for plotting.

```python
import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
```

The `_get_routes` function extracts the routes from the solution provided by the OR-Tools solver. It iterates over each vehicle and constructs the route by following the indices from the solution.

```python
def _get_routes(data, manager, routing, solution):
    routes = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(data['locations'][node_index])
            index = solution.Value(routing.NextVar(index))
        routes.append(route)
    return routes
```

The `DeliveryOptimizer` class encapsulates the VRP logic. The constructor initializes the number of vehicles, depot location, and lists for locations, time windows, demands, and vehicle capacities.

```python
class DeliveryOptimizer:
    def __init__(self, num_vehicles, depot):
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.locations = []
        self.time_windows = []
        self.demands = []
        self.vehicle_capacities = []
```

The `add_location` method adds a new location with its time window and demand to the respective lists.

```python
def add_location(self, location, time_window, demand):
    self.locations.append(location)
    self.time_windows.append(time_window)
    self.demands.append(demand)
```

The `_create_data_model` method constructs the data model required by the OR-Tools solver. It includes locations, number of vehicles, depot index, time windows, demands, vehicle capacities, and a distance matrix calculated using Euclidean distances.

```python
def _create_data_model(self):
    data = {}
    data['locations'] = [self.depot] + self.locations
    data['num_vehicles'] = self.num_vehicles
    data['depot'] = 0
    data['time_windows'] = [(0, 0)] + self.time_windows
    data['demands'] = [0] + self.demands
    data['vehicle_capacities'] = self.vehicle_capacities

    num_locations = len(data['locations'])
    data['distance_matrix'] = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            data['distance_matrix'][i][j] = np.linalg.norm(
                np.array(data['locations'][i]) - np.array(data['locations'][j]))

    return data
```

The `optimize_routes` method sets up and solves the VRP using OR-Tools. It defines distance and demand callbacks, adds capacity and time window constraints, and specifies the search parameters for the solver.

```python
def optimize_routes(self):
    data = self._create_data_model()
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')

    time = 'Time'
    routing.AddDimension(
        transit_callback_index, 30, 300, False, time)
    time_dimension = routing.GetDimensionOrDie(time)

    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        return _get_routes(data, manager, routing, solution)
    else:
        return None
```

The `visualize_routes` method uses `matplotlib` to plot the optimized routes. It creates a heatmap for demand intensity, plots the depot and delivery locations, and visualizes the routes with direction arrows and labels.

```python
def visualize_routes(self, routes):
    plt.figure(figsize=(15, 12))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", ["#FFF3E0", "#FF5722"], N=n_bins)

    x = np.array([loc[0] for loc in self.locations])
    y = np.array([loc[1] for loc in self.locations])
    z = np.array(self.demands)
    plt.hist2d(x, y, weights=z, bins=20, cmap=cmap, alpha=0.3)
    plt.colorbar(label='Demand Intensity')

    plt.plot(self.depot[0], self.depot[1], 'ko', markersize=15, label='Depot')

    locations = np.array(self.locations)
    scatter = plt.scatter(locations[:, 0], locations[:, 1], c=self.demands,
                          cmap='viridis', s=100, edgecolor='black', label='Delivery Locations')
    plt.colorbar(scatter, label='Demand')

    for i, route in enumerate(routes):
        route_arr = np.array(route)
        plt.plot(route_arr[:, 0], route_arr[:, 1], colors[i % len(colors)] + '-o',
                 linewidth=2, markersize=8, label=f'Vehicle {i + 1}')

        for j in range(len(route) - 1):
            plt.annotate('', xy=route[j + 1], xytext=route[j],
                         arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)], lw=2),
                         va='center', ha='center')

    for i, loc in enumerate(self.locations):
        plt.annotate(f'L{i + 1}', xy=(loc[0], loc[1]), xytext=(3, 3), textcoords='offset points')

    for loc, demand in zip(self.locations, self.demands):
        circle = Circle((loc[0], loc[1]), demand / 2, fill=False, color='r', linestyle='dashed')
        plt.gca().add_artist(circle)

    plt.title('Optimized Delivery Routes with Demand Heatmap')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```

In summary, this script defines a `DeliveryOptimizer` class to solve and visualize a vehicle routing problem using OR-Tools and `matplotlib`. The class includes methods to add locations, set vehicle capacities, create the data model, optimize routes, and visualize the results.

## Linear supply chain 
The selected code is a Python script that uses the Pyomo library to solve a linear optimization problem for supply chain management. The goal is to minimize the total cost of transporting products from warehouses to stores while satisfying supply and demand constraints.

First, the script imports necessary modules from Pyomo, including `ConcreteModel`, `Var`, `Objective`, `Constraint`, `NonNegativeReals`, `minimize`, and `value`. It also imports `SolverFactory` to solve the optimization model.

```python
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, minimize, value
from pyomo.opt import SolverFactory
```

The script then creates a `ConcreteModel` instance named `model`.

```python
model = ConcreteModel()
```

Next, it defines sets for warehouses, stores, and products. These sets represent the different entities involved in the supply chain.

```python
warehouses = ['Warehouse1', 'Warehouse2']
stores = ['Store1', 'Store2']
products = ['A', 'B']
```

The decision variables are defined using the `Var` function. The `model.transport` variable represents the quantity of each product transported from each warehouse to each store. The domain is set to `NonNegativeReals` to ensure non-negative values.

```python
model.transport = Var(warehouses, stores, products, domain=NonNegativeReals)
```

Parameters such as transportation costs, storage costs, supply limits, and demand requirements are defined using dictionaries. These parameters provide the necessary data for the optimization model.

```python
costs = {('Warehouse1', 'Store1'): 2, ('Warehouse1', 'Store2'): 4, ('Warehouse2', 'Store1'): 3, ('Warehouse2', 'Store2'): 2}
storage_costs = {'A': 1, 'B': 1.5}
supply_limits = {('Warehouse1', 'A'): 100, ('Warehouse1', 'B'): 150, ('Warehouse2', 'A'): 200, ('Warehouse2', 'B'): 100}
demand_requirements = {('Store1', 'A'): 80, ('Store1', 'B'): 100, ('Store2', 'A'): 120, ('Store2', 'B'): 150}
```

The objective function is defined using the `Objective` function. The `objective_rule` function calculates the total cost, which includes transportation and storage costs. The objective is to minimize this total cost.

```python
def objective_rule(model):
    return sum(model.transport[w, s, p] * (costs[(w, s)] + storage_costs[p]) for w in warehouses for s in stores for p in products)

model.objective = Objective(rule=objective_rule, sense=minimize)
```

Supply constraints ensure that the total quantity of each product transported from each warehouse does not exceed the supply limits. These constraints are defined using the `Constraint` function and the `supply_rule` function.

```python
def supply_rule(model, w, p):
    return sum(model.transport[w, s, p] for s in stores) <= supply_limits[(w, p)]

model.supply_constraint = Constraint(warehouses, products, rule=supply_rule)
```

Demand constraints ensure that the total quantity of each product transported to each store meets the demand requirements. These constraints are defined using the `Constraint` function and the `demand_rule` function.

```python
def demand_rule(model, s, p):
    return sum(model.transport[w, s, p] for w in warehouses) == demand_requirements[(s, p)]

model.demand_constraint = Constraint(stores, products, rule=demand_rule)
```

The model is solved using the `SolverFactory` with the 'glpk' solver. The results are printed and saved to a file named `results.txt`.

```python
solver = SolverFactory('glpk')
solver.solve(model)

for w in warehouses:
    for s in stores:
        for p in products:
            print(f'Transport {p} from {w} to {s}:', value(model.transport[w, s, p]))

print("Total Cost =", value(model.objective))

with open('results.txt', 'w') as f:
    for w in warehouses:
        for s in stores:
            for p in products:
                f.write(f'Transport {p} from {w} to {s}: {value(model.transport[w, s, p])}\n')
    f.write(f'Total Cost = {value(model.objective)}\n')
```

In summary, this script sets up and solves a linear optimization problem to minimize transportation and storage costs in a supply chain while satisfying supply and demand constraints.

## Non convex optimization energy dispatch
The selected code is a Python script that uses the Pyomo library to solve a non-convex optimization problem for energy dispatch. The goal is to minimize the total cost of energy production from various sources while meeting the demand and adhering to generation limits.

First, the script imports necessary modules from Pyomo and Plotly. Pyomo is used for defining and solving the optimization model, while Plotly is used for visualizing the results.

```python
from pyomo.environ import *
import plotly.graph_objects as go
```

The `EnergyDispatch` class is defined to encapsulate the optimization model. The constructor initializes the sources, cost functions, generation limits, and demand. It also creates a `ConcreteModel` instance.

```python
class EnergyDispatch:
    def __init__(self, sources, cost_functions, generation_limits, demand):
        self.sources = sources
        self.cost_functions = cost_functions
        self.generation_limits = generation_limits
        self.demand = demand

        self.model = ConcreteModel()
```

Decision variables are defined using the `Var` function, representing the amount of energy produced by each source. The domain is set to `NonNegativeReals` to ensure non-negative values.

```python
self.model.energy = Var(self.sources, domain=NonNegativeReals)
```

The objective function is defined using the `Objective` function. The `cost_rule` method calculates the total cost by summing the cost functions for each source. The objective is to minimize this total cost.

```python
self.model.cost = Objective(rule=self.cost_rule, sense=minimize)
```

Constraints are added to the model to ensure that the total energy produced meets the demand and that the energy produced by each source is within its generation limits. These constraints are defined using the `Constraint` function and the `demand_rule`, `generation_limit_rule_lower`, and `generation_limit_rule_upper` methods.

```python
self.model.demand_constraint = Constraint(rule=self.demand_rule)
self.model.generation_limit_constraint_lower = Constraint(self.sources, rule=self.generation_limit_rule_lower)
self.model.generation_limit_constraint_upper = Constraint(self.sources, rule=self.generation_limit_rule_upper)
```

The `solve` method sets up and solves the optimization model using a solver. The results are printed and visualized using Plotly. The energy produced by each source is displayed in a bar chart.

```python
def solve(self):
    solver = SolverFactory('')
    solver.solve(self.model)

    energy_produced = []
    for src in self.sources:
        print(f"Energy produced from {src}: {self.model.energy[src].value}")
        energy_produced.append(self.model.energy[src].value)

    fig = go.Figure(data=[go.Bar(x=self.sources, y=energy_produced)])
    fig.update_layout(title_text='Energy Produced from Each Source')
    fig.show()
```

In summary, this script defines an `EnergyDispatch` class to solve a non-convex optimization problem for energy dispatch using Pyomo. The class includes methods to set up the model, define the objective function and constraints, solve the model, and visualize the results.

## Non linear optimization for manufacturing
The selected code is a Python script that visualizes production rates and resource allocations for a manufacturing optimization model using `matplotlib`. The script imports the necessary modules and data from `Nonlinear_Optimization`, which includes the optimization model and machine data.

First, the script defines a list of products, `products`, which includes 'P1' and 'P2'. It then prepares the data for visualization by extracting the production rates for each product from the optimization model. This is done using a list comprehension that iterates over the products and retrieves the production values.

```python
products = ['P1', 'P2']
production_rates = [model.production[p].value for p in products]
```

Next, the script calculates the total resources allocated to each resource type ('R1' and 'R2'). This is achieved using a dictionary comprehension that sums the resource usage across all products and machines.

```python
resources_allocated = {r: sum(model.resource_use[p, m].value for p in products for m in machines) for r in ['R1', 'R2']}
```

The script then creates a bar chart to visualize the production rates per product. It sets the figure size, plots the bar chart with the production rates, and labels the axes and title.

```python
plt.figure(figsize=(8, 6))
plt.bar(products, production_rates, color='skyblue')
plt.title('Production Rates per Product')
plt.xlabel('Products')
plt.ylabel('Production Rate')
plt.show()
```

Following this, the script generates a pie chart to visualize the resource allocations. It sets the figure size, plots the pie chart with the resource allocation data, and ensures the pie chart is circular by setting the aspect ratio to equal.

```python
plt.figure(figsize=(8, 6))
plt.pie(resources_allocated.values(), labels=resources_allocated.keys(), autopct='%1.1f%%', startangle=140)
plt.title('Resource Allocations')
plt.axis('equal')
plt.show()
```

In summary, this script visualizes the production rates and resource allocations from a manufacturing optimization model using bar and pie charts, making it easier to understand the distribution of production and resource usage.

## Quadratic optimization for portfolio management
The selected code is a Python script that uses the Pyomo library to solve a quadratic optimization problem for investment portfolio selection. The goal is to minimize the variance of the portfolio while achieving a target return.

First, the script imports necessary modules from Pyomo and NumPy. Pyomo is used for defining and solving the optimization model, while NumPy is used for handling arrays and matrices.

```python
from pyomo.environ import *
import numpy as np
from pyomo.opt import SolverFactory, ResultsFormat
```

The script defines example data, including the number of investments (`n`), expected returns (`returns`), covariance matrix (`cov_matrix`), and target return (`target_return`).

```python
n = 4  # Number of investments
returns = np.array([0.12, 0.10, 0.07, 0.03])  # Expected returns
cov_matrix = np.array([[0.10, 0.01, 0.02, 0.00],  # Covariance matrix
                       [0.01, 0.08, 0.01, 0.00],
                       [0.02, 0.01, 0.07, 0.00],
                       [0.00, 0.00, 0.00, 0.02]])
target_return = 0.08  # Target return
```

A `ConcreteModel` instance named `model` is created to represent the optimization model.

```python
model = ConcreteModel()
```

Decision variables are defined using the `Var` function. The `model.x` variable represents the proportion of the total investment allocated to each investment option. The domain is set to `NonNegativeReals` to ensure non-negative values.

```python
model.x = Var(range(n), domain=NonNegativeReals)
```

The objective function is defined using the `Objective` function. The `portfolio_variance` function calculates the total variance of the portfolio by summing the products of the covariance matrix elements and the decision variables. The objective is to minimize this variance.

```python
def portfolio_variance(model):
    return sum(cov_matrix[i, j] * model.x[i] * model.x[j] for i in range(n) for j in range(n))

model.objective = Objective(rule=portfolio_variance, sense=minimize)
```

Constraints are added to the model to ensure that the total return meets the target return and that the sum of the investment proportions equals 1. These constraints are defined using the `Constraint` function and the `return_constraint` and `sum_constraint` functions.

```python
def return_constraint(model):
    return sum(returns[i] * model.x[i] for i in range(n)) >= target_return

model.return_constraint = Constraint(rule=return_constraint)

def sum_constraint(model):
    return sum(model.x[i] for i in range(n)) == 1

model.sum_constraint = Constraint(rule=sum_constraint)
```

The model is solved using the `SolverFactory` with the 'ipopt' solver. The results, including the investment proportions, variance, and return, are printed and saved to a file named `results.yml`.

```python
solver = SolverFactory('ipopt')
results = solver.solve(model)

investment_proportions = [model.x[i].value for i in range(n)]
print("Investment Proportions:", investment_proportions)
print("Variance:", model.objective)
print("Return:", sum(returns[i] * investment_proportions[i] for i in range(n)))

results.write(filename='results.yml', format=ResultsFormat.yaml)
```

In summary, this script sets up and solves a quadratic optimization problem to minimize the variance of an investment portfolio while achieving a specified target return. The results include the optimal investment proportions, the minimized variance, and the achieved return.

## Scheduling integer programming
The selected code is a Python script that uses the Pyomo library to solve an integer programming problem for employee shift scheduling. The goal is to minimize staffing costs while meeting shift requirements and respecting employee availability.

First, the script imports necessary modules from Pyomo and the `pickle` module for saving results. It defines example data, including lists of shifts and employees, and dictionaries for availability, shift requirements, and staffing costs.

```python
from pyomo.environ import *
import pickle

shifts = ['Morning', 'Afternoon', 'Night']
employees = ['Alice', 'Bob', 'Charlie']
availability = {('Alice', 'Morning'): 1, ('Alice', 'Afternoon'): 0, ('Alice', 'Night'): 1,
                ('Bob', 'Morning'): 1, ('Bob', 'Afternoon'): 1, ('Bob', 'Night'): 0,
                ('Charlie', 'Morning'): 0, ('Charlie', 'Afternoon'): 1, ('Charlie', 'Night'): 1}
shift_requirements = {'Morning': 1, 'Afternoon': 2, 'Night': 1}
staffing_costs = {('Alice', 'Morning'): 100, ('Alice', 'Night'): 150,
                  ('Bob', 'Morning'): 100, ('Bob', 'Afternoon'): 100,
                  ('Charlie', 'Afternoon'): 120, ('Charlie', 'Night'): 120}
```

A `ConcreteModel` instance named `model` is created to represent the optimization model. Decision variables are defined using the `Var` function, representing whether an employee is working a particular shift. The domain is set to `Binary` to ensure values are either 0 or 1.

```python
model = ConcreteModel()
model.working = Var(employees, shifts, domain=Binary)
```

The objective function is defined using the `Objective` function. The `objective_rule` function calculates the total staffing cost by summing the products of the decision variables and the staffing costs. The objective is to minimize this total cost.

```python
def objective_rule(model):
    return sum(model.working[e, s] * staffing_costs.get((e, s), 0) for e in employees for s in shifts)

model.objective = Objective(rule=objective_rule, sense=minimize)
```

Constraints are added to the model to ensure that the number of employees working each shift meets the shift requirements and that employees only work shifts they are available for. These constraints are defined using the `Constraint` function and the `shift_requirement_rule` and `availability_rule` functions.

```python
def shift_requirement_rule(model, s):
    return sum(model.working[e, s] for e in employees) >= shift_requirements[s]

model.shift_requirement = Constraint(shifts, rule=shift_requirement_rule)

def availability_rule(model, e, s):
    return model.working[e, s] <= availability.get((e, s), 0)

model.availability = Constraint(employees, shifts, rule=availability_rule)
```

The model is solved using the `SolverFactory` with the 'glpk' solver. The results, including the schedule, are printed and saved to a file named `schedule_data.pkl`.

```python
solver = SolverFactory('glpk')
solver.solve(model)

schedule = {(e, s): model.working[e, s].value for e in employees for s in shifts}
print("Schedule:", schedule)

with open('schedule_data.pkl', 'wb') as f:
    pickle.dump((schedule, shifts, employees), f)
```

In summary, this script sets up and solves an integer programming problem to minimize staffing costs while meeting shift requirements and respecting employee availability. The results include the optimal schedule for employees.

## Stochastic optimization for inventory management
The selected code is a Python script that uses the Pyomo library to solve a stochastic optimization problem for inventory management. The goal is to minimize the total cost, which includes holding, ordering, and shortage costs, while considering storage capacity and demand scenarios.

First, the script imports necessary modules from Pyomo, NumPy, SciPy, Matplotlib, and Plotly. Pyomo is used for defining and solving the optimization model, NumPy and SciPy for handling arrays and statistical functions, and Matplotlib and Plotly for visualizing the results.

```python
from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Objective, minimize, Constraint
from pyomo.environ import SolverFactory
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
```

The `InventoryManagement` class is defined to encapsulate the optimization model. The constructor initializes the products, storage capacity, demand scenarios, and cost parameters. It also creates a `ConcreteModel` instance.

```python
class InventoryManagement:
    def __init__(self, products, storage_capacity, demand_scenarios, holding_cost_per_unit, ordering_cost_per_order, shortage_cost_per_unit):
        self.products = products
        self.storage_capacity = storage_capacity
        self.demand_scenarios = demand_scenarios
        self.holding_cost_per_unit = holding_cost_per_unit
        self.ordering_cost_per_order = ordering_cost_per_order
        self.shortage_cost_per_unit = shortage_cost_per_unit

        self.model = ConcreteModel()
```

Decision variables are defined using the `Var` function, representing the reorder point and order quantity for each product. The domain is set to `NonNegativeReals` to ensure non-negative values.

```python
self.model.reorder_point = Var(self.products, domain=NonNegativeReals)
self.model.order_quantity = Var(self.products, domain=NonNegativeReals)
```

The objective function is defined using the `Objective` function. The `expected_cost_rule` method calculates the total cost, which includes shortage costs, ordering costs, and holding costs. The objective is to minimize this total cost.

```python
self.model.total_cost = Objective(rule=self.expected_cost_rule, sense=minimize)
```

Constraints are added to the model to ensure that the total inventory does not exceed the storage capacity. This constraint is defined using the `Constraint` function and the `storage_capacity_rule` method.

```python
self.model.storage_capacity_constraint = Constraint(rule=self.storage_capacity_rule)
```

The `solve` method sets up and solves the optimization model using a solver. The results, including the reorder points and order quantities, are printed and visualized using Plotly and Matplotlib.

```python
def solve(self):
    solver = SolverFactory('ipopt')
    solver.solve(self.model)

    for product in self.products:
        print(f"Reorder point for {product}: {self.model.reorder_point[product].value}")
        print(f"Order quantity for {product}: {self.model.order_quantity[product].value}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=self.demand_scenarios, y=[norm.cdf(x, 100, 20) for x in self.demand_scenarios], mode='markers', name='Demand CDF'))
    fig.add_trace(go.Scatter(x=[self.model.reorder_point[p].value for p in self.products], y=[norm.cdf(self.model.reorder_point[p].value, 100, 20) for p in self.products], mode='markers', name='Reorder Point'))
    fig.update_layout(title='Demand CDF vs Reorder Point', xaxis_title='Demand', yaxis_title='Probability')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[self.model.reorder_point[p].value for p in self.products], y=[self.model.order_quantity[p].value for p in self.products], mode='markers', name='Order Quantity'))
    fig.update_layout(title='Order Quantity vs Reorder Point', xaxis_title='Reorder Point', yaxis_title='Order Quantity')
    fig.show()

    plt.scatter(self.demand_scenarios, [norm.cdf(x, 100, 20) for x in self.demand_scenarios], label='Demand CDF')
    plt.scatter([self.model.reorder_point[p].value for p in self.products], [norm.cdf(self.model.reorder_point[p].value, 100, 20) for p in self.products], label='Reorder Point')
    plt.xlabel('Demand')
    plt.ylabel('Probability')
    plt.title('Demand CDF vs Reorder Point')
    plt.legend()

    plt.scatter([self.model.reorder_point[p].value for p in self.products], [self.model.order_quantity[p].value for p in self.products], label='Order Quantity')
    plt.xlabel('Reorder Point')
    plt.ylabel('Order Quantity')
    plt.title('Order Quantity vs Reorder Point')
    plt.legend()
    plt.show()
```

In summary, this script defines an `InventoryManagement` class to solve a stochastic optimization problem for inventory management using Pyomo. The class includes methods to set up the model, define the objective function and constraints, solve the model, and visualize the results.

## Conclusion
The selected code snippets demonstrate the application of optimization techniques in various domains, including linear programming, integer programming, quadratic optimization, and stochastic optimization. These optimization models address real-world problems such as supply chain management, vehicle routing, energy dispatch, employee scheduling, portfolio management, and inventory management. By using optimization libraries like GAMS, Pyomo, and OR-Tools, these scripts provide efficient solutions to complex decision-making problems, helping organizations optimize their resources, reduce costs, and improve operational efficiency. The visualization components in the scripts enhance the understanding and interpretation of the optimization results, making them more accessible to stakeholders. Overall, these code snippets showcase the versatility and power of optimization techniques in solving diverse optimization problems across different industries.
and as we can see here in the code snippets, the optimization techniques are used in different domains and they are very helpful in solving real-world problems.
if you want to see the results of the code snippets you can run the code snippets in your local machine or you can contact me to provide you with the results of the code snippets.
and my results are in the folder of the respective problem.