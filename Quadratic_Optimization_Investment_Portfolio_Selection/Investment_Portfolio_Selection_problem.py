from pyomo.environ import *
import numpy as np
from pyomo.opt import SolverFactory, ResultsFormat

# Example data
n = 4  # Number of investments
returns = np.array ([0.12, 0.10, 0.07, 0.03])  # Expected returns
cov_matrix = np.array ([[0.10, 0.01, 0.02, 0.00],  # Covariance matrix
                        [0.01, 0.08, 0.01, 0.00],
                        [0.02, 0.01, 0.07, 0.00],
                        [0.00, 0.00, 0.00, 0.02]])
target_return = 0.08  # Target return

# Pyomo model
model = ConcreteModel ()

# Decision Variables
model.x = Var (range (n), domain=NonNegativeReals)


# Objective: Minimize Variance
def portfolio_variance(model):
    return sum (cov_matrix [i, j] * model.x [i] * model.x [j] for i in range (n) for j in range (n))


model.objective = Objective (rule=portfolio_variance, sense=minimize)


# Constraints
def return_constraint(model):
    return sum (returns [i] * model.x [i] for i in range (n)) >= target_return


model.return_constraint = Constraint (rule=return_constraint)


def sum_constraint(model):
    return sum (model.x [i] for i in range (n)) == 1


model.sum_constraint = Constraint (rule=sum_constraint)

# Solver
solver = SolverFactory ('ipopt')
results = solver.solve (model)

# Results
investment_proportions = [model.x [i].value for i in range (n)]
print ("Investment Proportions:", investment_proportions)
print ("Variance:", model.objective)
print ("Return:", sum (returns [i] * investment_proportions [i] for i in range (n)))

# save results and model
results.write(filename='results.yml', format=ResultsFormat.yaml)
