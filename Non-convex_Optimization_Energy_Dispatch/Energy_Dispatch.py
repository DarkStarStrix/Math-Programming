from pyomo.environ import *
import plotly.graph_objects as go


class EnergyDispatch:
    def __init__(self, sources, cost_functions, generation_limits, demand):
        self.sources = sources
        self.cost_functions = cost_functions
        self.generation_limits = generation_limits
        self.demand = demand

        self.model = ConcreteModel ()

        # Decision Variables
        self.model.energy = Var (self.sources, domain=NonNegativeReals)

        # Objective Function
        self.model.cost = Objective (rule=self.cost_rule, sense=minimize)

        # Constraints
        self.model.demand_constraint = Constraint (rule=self.demand_rule)
        self.model.generation_limit_constraint_lower = Constraint (self.sources, rule=self.generation_limit_rule_lower)
        self.model.generation_limit_constraint_upper = Constraint (self.sources, rule=self.generation_limit_rule_upper)

    def cost_rule(self, model):
        return sum (self.cost_functions [src] (model.energy [src]) for src in self.sources)

    def demand_rule(self, model):
        return sum (model.energy [src] for src in self.sources) == self.demand

    def generation_limit_rule_lower(self, model, src):
        return self.generation_limits [src] [0] <= model.energy [src]

    def generation_limit_rule_upper(self, model, src):
        return model.energy [src] <= self.generation_limits [src] [1]

    def solve(self):
        # Solver
        solver = SolverFactory (
            '')  # a non_convex solver is required I can't find a compatible one for my system and the other one I found is not free
        solver.solve (self.model)

        # Results
        energy_produced = []
        for src in self.sources:
            print (f"Energy produced from {src}: {self.model.energy [src].value}")
            energy_produced.append (self.model.energy [src].value)

        # Plotting using Plotly
        fig = go.Figure (data=[go.Bar (x=self.sources, y=energy_produced)])
        fig.update_layout (title_text='Energy Produced from Each Source')
        fig.show ()


# Example data
sources = ['Coal', 'Gas', 'Wind', 'Solar', 'Hydro']
cost_functions = {'Coal': lambda x: x ** 2, 'Gas': lambda x: x ** 1.5, 'Wind': lambda x: x, 'Solar': lambda x: x,
                  'Hydro': lambda x: x ** 1.2}
generation_limits = {'Coal': (50, 200), 'Gas': (30, 150), 'Wind': (0, 100), 'Solar': (0, 80), 'Hydro': (20, 100)}
demand = 300

# Create an instance of the EnergyDispatch class
energy_dispatch = EnergyDispatch (sources, cost_functions, generation_limits, demand)

# Solve the model
energy_dispatch.solve ()
