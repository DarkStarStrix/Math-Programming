from Nonlinear_Optimization import model, machines
import matplotlib.pyplot as plt

products = ['P1', 'P2']
# Data preparation for visualization
production_rates = [model.production[p].value for p in products]
resources_allocated = {r: sum(model.resource_use[p, m].value for p in products for m in machines) for r in ['R1', 'R2']}

# Bar chart for production rates
plt.figure(figsize=(8, 6))
plt.bar(products, production_rates, color='skyblue')
plt.title('Production Rates per Product')
plt.xlabel('Products')
plt.ylabel('Production Rate')
plt.show()

# Pie chart for resource allocations
plt.figure(figsize=(8, 6))
plt.pie(resources_allocated.values(), labels=resources_allocated.keys(), autopct='%1.1f%%', startangle=140)
plt.title('Resource Allocations')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()
