from pyomo.environ import *

# Example data
locations = ['Loc1', 'Loc2', 'Loc3']
costs = {'Loc1': 1000, 'Loc2': 1500, 'Loc3': 2000}
coverage_areas = {'Loc1': 500, 'Loc2': 800, 'Loc3': 1200}
budget = 3000

# Pyomo model
model = ConcreteModel ()

# Decision Variables
model.build_node = Var (locations, domain=Binary)
model.node_capacity = Var (locations, domain=NonNegativeReals, initialize=lambda model, loc: costs [loc])


# Objective: Maximize network coverage
def coverage_rule(model):
    return sum (coverage_areas [loc] * model.build_node [loc] for loc in locations)


model.coverage = Objective (rule=coverage_rule, sense=maximize)


# Constraints
def capacity_rule(model, loc):
    return model.node_capacity [loc] >= coverage_areas [loc] * model.build_node [loc]


model.capacity_constraint = Constraint (locations, rule=capacity_rule)


def budget_rule(model):
    return sum (costs [loc] * model.build_node [loc] for loc in locations) <= budget


model.budget_constraint = Constraint (rule=budget_rule)

# Solver
solver = SolverFactory ('glpk')
solver.solve (model)

# Results
for loc in locations:
    print (f"Build node at {loc}: {model.build_node [loc].value}")
    print (f"Node capacity at {loc}: {model.node_capacity [loc].value}")
