from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, minimize, value
from pyomo.opt import SolverFactory

# Model
model = ConcreteModel ()

# Sets
warehouses = ['Warehouse1', 'Warehouse2']
stores = ['Store1', 'Store2']
products = ['A', 'B']

# Decision Variables
model.transport = Var (warehouses, stores, products, domain=NonNegativeReals)

# Parameters (example values)
costs = {('Warehouse1', 'Store1'): 2, ('Warehouse1', 'Store2'): 4,
         ('Warehouse2', 'Store1'): 3, ('Warehouse2', 'Store2'): 2}
storage_costs = {'A': 1, 'B': 1.5}
supply_limits = {('Warehouse1', 'A'): 100, ('Warehouse1', 'B'): 150,
                 ('Warehouse2', 'A'): 200, ('Warehouse2', 'B'): 100}
demand_requirements = {('Store1', 'A'): 80, ('Store1', 'B'): 100,
                       ('Store2', 'A'): 120, ('Store2', 'B'): 150}


# Objective Function
def objective_rule(model):
    return sum (model.transport [w, s, p] * (costs [(w, s)] + storage_costs [p])
                for w in warehouses for s in stores for p in products)


model.objective = Objective (rule=objective_rule, sense=minimize)


# Constraints
# Supply constraints
def supply_rule(model, w, p):
    return sum (model.transport [w, s, p] for s in stores) <= supply_limits [(w, p)]


model.supply_constraint = Constraint (warehouses, products, rule=supply_rule)


# Demand constraints
def demand_rule(model, s, p):
    return sum (model.transport [w, s, p] for w in warehouses) == demand_requirements [(s, p)]


model.demand_constraint = Constraint (stores, products, rule=demand_rule)

# Solving the model
solver = SolverFactory ('glpk')
solver.solve (model)

# Output results
for w in warehouses:
    for s in stores:
        for p in products:
            print (f'Transport {p} from {w} to {s}:', value (model.transport [w, s, p]))

print ("Total Cost =", value (model.objective))

# save the results
with open ('results.txt', 'w') as f:
    for w in warehouses:
        for s in stores:
            for p in products:
                f.write (f'Transport {p} from {w} to {s}: {value (model.transport [w, s, p])}\n')

    f.write (f'Total Cost = {value (model.objective)}\n')
