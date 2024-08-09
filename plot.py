import matplotlib.pyplot as plt
import numpy as np

# Data from the image
categories = ['Never', 'A few times', 'Very often', 'I do not know']
male_values = [83, 10, 4, 3]
female_values = [77, 11, 6, 5]

# Define bar positions
y = np.arange(len(categories))

# Bar width
bar_width = 0.35

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the data
bars1 = ax.barh(y - bar_width/2, male_values, bar_width, label='Male', color='steelblue')
bars2 = ax.barh(y + bar_width/2, female_values, bar_width, label='Female', color='coral')

# Add text annotations
for bar in bars1:
    width = bar.get_width()
    ax.annotate(f'{width}%',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),  # 3 points horizontal offset
                textcoords="offset points",
                ha='left', va='center')

for bar in bars2:
    width = bar.get_width()
    ax.annotate(f'{width}%',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),  # 3 points horizontal offset
                textcoords="offset points",
                ha='left', va='center')

# Set labels and title
ax.set_xlabel('Percentage')
ax.set_yticks(y)
ax.set_yticklabels(categories)
ax.legend()

# Display the plot
plt.tight_layout()
plt.savefig('plots/reproduced_graph.png', format='png')
plt.show()