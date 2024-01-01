import seaborn as sns
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load variables from a file
with open('schedule_data.pkl', 'rb') as f:
    schedule, shifts, employees = pickle.load(f)

# Create a heatmap-style table
data = np.array([[schedule[(e, s)] for s in shifts] for e in employees])

# Create a DataFrame for better visualization
df = pd.DataFrame(data, index=employees, columns=shifts)

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap='Greens', fmt='g', cbar=True)

# Display the heatmap
plt.title('Employee Shift Schedule')
plt.xlabel('Shifts')
plt.ylabel('Employees')
plt.show()
