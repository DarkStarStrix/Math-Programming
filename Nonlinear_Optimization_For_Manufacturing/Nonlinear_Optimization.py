from pyomo.environ import *

# Example data (simplified for demonstration)
machines = ['M1', 'M2']
products = ['P1', 'P2']
machine_capacity = {'M1': 100, 'M2': 150}
resource_availability = {'R1': 200, 'R2': 300}
quality_standards = {'P1': 0.9, 'P2': 0.95}
production_costs = {'P1': 5, 'P2': 7}  # Cost per unit
resource_usage = {('P1', 'M1'): 1, ('P1', 'M2'): 0, ('P2', 'M1'): 0,
                  ('P2', 'M2'): 1}  # Resource usage per product and machine

# Pyomo model
model = ConcreteModel ()

# Decision Variables
model.production = Var (products, domain=NonNegativeReals)
model.resource_use = Var (products, machines, domain=NonNegativeReals)


# Objective: Minimize production cost
def cost_rule(model):
    return sum (production_costs [p] * model.production [p] for p in products)


model.cost = Objective (rule=cost_rule, sense=minimize)


# Constraints
def capacity_rule(model, m):
    return sum (model.resource_use [p, m] for p in products) <= machine_capacity [m]


model.machine_capacity = Constraint (machines, rule=capacity_rule)


def quality_rule(model, p):
    return sum (model.resource_use [p, m] for m in machines) >= quality_standards [p] * model.production [p]


model.quality_standards = Constraint (products, rule=quality_rule)


def resource_rule(model, r):
    return sum (resource_usage [p, m] * model.resource_use [p, m] for p in products for m in machines) <= \
        resource_availability [r]


model.resource_availability = Constraint (['R1', 'R2'], rule=resource_rule)

# Add a lower bound to the production decision variable
for p in products:
    model.production [p].setlb (1)

# Solver
solver = SolverFactory ('ipopt')
results = solver.solve (model)

# Check solver status and termination condition
print ('Solver Status:', results.solver.status)
print ('Termination Condition:', results.solver.termination_condition)

# Results
for p in products:
    print (f"Production rate of {p}: {model.production [p].value}")
