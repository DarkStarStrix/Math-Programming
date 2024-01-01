from pyomo.environ import *
import pickle

# Example data
shifts = ['Morning', 'Afternoon', 'Night']
employees = ['Alice', 'Bob', 'Charlie']
availability = {('Alice', 'Morning'): 1, ('Alice', 'Afternoon'): 0, ('Alice', 'Night'): 1,
                ('Bob', 'Morning'): 1, ('Bob', 'Afternoon'): 1, ('Bob', 'Night'): 0,
                ('Charlie', 'Morning'): 0, ('Charlie', 'Afternoon'): 1, ('Charlie', 'Night'): 1}
shift_requirements = {'Morning': 1, 'Afternoon': 2, 'Night': 1}
staffing_costs = {('Alice', 'Morning'): 100, ('Alice', 'Night'): 150,
                  ('Bob', 'Morning'): 100, ('Bob', 'Afternoon'): 100,
                  ('Charlie', 'Afternoon'): 120, ('Charlie', 'Night'): 120}

# Pyomo model
model = ConcreteModel ()

# Decision Variables
model.working = Var (employees, shifts, domain=Binary)


# Objective: Minimize staffing cost
def objective_rule(model):
    return sum (model.working [e, s] * staffing_costs.get ((e, s), 0) for e in employees for s in shifts)


model.objective = Objective (rule=objective_rule, sense=minimize)


# Constraints
def shift_requirement_rule(model, s):
    return sum (model.working [e, s] for e in employees) >= shift_requirements [s]


model.shift_requirement = Constraint (shifts, rule=shift_requirement_rule)


def availability_rule(model, e, s):
    return model.working [e, s] <= availability.get ((e, s), 0)


model.availability = Constraint (employees, shifts, rule=availability_rule)

# Solver
solver = SolverFactory ('glpk')
solver.solve (model)

# Results
schedule = {(e, s): model.working [e, s].value for e in employees for s in shifts}
print ("Schedule:", schedule)

# Save variables to a file
with open ('schedule_data.pkl', 'wb') as f:
    pickle.dump ((schedule, shifts, employees), f)
