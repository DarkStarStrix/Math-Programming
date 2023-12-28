import matplotlib.pyplot as plt

# Investment proportions and other results from the optimization
investment_proportions = [0.27239456962864383, 0.2849979144012793, 0.13836565633482306, 0.3042418596352539]
investment_labels = ['Investment 1', 'Investment 2', 'Investment 3', 'Investment 4']
expected_return = 0.07999999152806042

# Pie chart for investment proportions
plt.figure(figsize=(8, 6))
plt.pie(investment_proportions, labels=investment_labels, autopct='%1.1f%%', startangle=140)
plt.title('Portfolio Investment Proportions')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

# Show the pie chart
plt.show()

# Bar graph for expected return
plt.figure(figsize=(6, 4))
plt.bar(['Expected Return'], [expected_return], color='green')
plt.title('Portfolio Expected Return')
plt.ylabel('Return')

# Show the bar graph
plt.show()

# Bar graph for investment proportions
plt.figure(figsize=(8, 6))
plt.bar(investment_labels, investment_proportions, color='blue')
plt.title('Portfolio Investment Proportions')
plt.ylabel('Proportion')

# Show the bar graph
plt.show()
