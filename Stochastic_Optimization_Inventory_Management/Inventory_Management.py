from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Objective, minimize, Constraint
from pyomo.environ import SolverFactory
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class InventoryManagement:
    def __init__(self, products, storage_capacity, demand_scenarios, holding_cost_per_unit, ordering_cost_per_order,
                 shortage_cost_per_unit):
        self.products = products
        self.storage_capacity = storage_capacity
        self.demand_scenarios = demand_scenarios
        self.holding_cost_per_unit = holding_cost_per_unit
        self.ordering_cost_per_order = ordering_cost_per_order
        self.shortage_cost_per_unit = shortage_cost_per_unit

        self.model = ConcreteModel ()

        # Decision Variables
        self.model.reorder_point = Var (self.products, domain=NonNegativeReals)
        self.model.order_quantity = Var (self.products, domain=NonNegativeReals)

        # Objective Function
        self.model.total_cost = Objective (rule=self.expected_cost_rule, sense=minimize)

        # Constraints
        self.model.storage_capacity_constraint = Constraint (rule=self.storage_capacity_rule)

    def expected_cost_rule(self, model):
        return sum (model.reorder_point [p] * self.shortage_cost_per_unit * (
            norm.cdf (model.reorder_point [p].value, 100, 20) if model.reorder_point [p].value is not None else 0) +
                    model.order_quantity [p] * self.ordering_cost_per_order * model.order_quantity [
                        p] +  # Changed this line
                    model.order_quantity [p] * self.holding_cost_per_unit * (
                        norm.cdf (model.reorder_point [p].value, 100, 20) if model.reorder_point [
                                                                                 p].value is not None else 0)
                    for p in self.products)

    def storage_capacity_rule(self, model):
        return sum (model.reorder_point [p] + model.order_quantity [p] for p in self.products) <= self.storage_capacity

    def solve(self):
        # Solver
        solver = SolverFactory ('ipopt')  # Hypothetical solver, actual may vary
        solver.solve (self.model)

        # Results
        for product in self.products:
            print (f"Reorder point for {product}: {self.model.reorder_point [product].value}")
            print (f"Order quantity for {product}: {self.model.order_quantity [product].value}")

        # Plotting using Plotly
        fig = go.Figure ()
        fig.add_trace (go.Scatter (x=self.demand_scenarios, y=[norm.cdf (x, 100, 20) for x in self.demand_scenarios],
                                   mode='markers', name='Demand CDF'))
        fig.add_trace (go.Scatter (x=[self.model.reorder_point [p].value for p in self.products],
                                   y=[norm.cdf (self.model.reorder_point [p].value, 100, 20) for p in
                                      self.products],
                                   mode='markers', name='Reorder Point'))
        fig.update_layout (title='Demand CDF vs Reorder Point', xaxis_title='Demand', yaxis_title='Probability')
        fig.show ()

        # order quantity vs reorder point
        fig = go.Figure ()
        fig.add_trace (go.Scatter (x=[self.model.reorder_point [p].value for p in self.products],
                                   y=[self.model.order_quantity [p].value for p in self.products],
                                   mode='markers', name='Order Quantity'))
        fig.update_layout (title='Order Quantity vs Reorder Point', xaxis_title='Reorder Point',
                           yaxis_title='Order Quantity')
        fig.show ()

        # using matplotlib
        plt.scatter (self.demand_scenarios, [norm.cdf (x, 100, 20) for x in self.demand_scenarios], label='Demand CDF')
        plt.scatter ([self.model.reorder_point [p].value for p in self.products],
                     [norm.cdf (self.model.reorder_point [p].value, 100, 20) for p in self.products],
                     label='Reorder Point')
        plt.xlabel ('Demand')
        plt.ylabel ('Probability')
        plt.title ('Demand CDF vs Reorder Point')
        plt.legend ()

        plt.scatter ([self.model.reorder_point [p].value for p in self.products],
                     [self.model.order_quantity [p].value for p in self.products],
                     label='Order Quantity')
        plt.xlabel ('Reorder Point')
        plt.ylabel ('Order Quantity')
        plt.title ('Order Quantity vs Reorder Point')
        plt.legend ()
        plt.show ()


# Example data
products = ['Product1', 'Product2']
storage_capacity = 2000
demand_scenarios = np.random.normal (200, 20, 100)  # Hypothetical demand scenarios
holding_cost_per_unit = 1
ordering_cost_per_order = 100
shortage_cost_per_unit = 1

# Create an instance of the InventoryManagement class
inventory_management = InventoryManagement (products, storage_capacity, demand_scenarios, holding_cost_per_unit,
                                            ordering_cost_per_order, shortage_cost_per_unit)

# Solve the model
inventory_management.solve ()
