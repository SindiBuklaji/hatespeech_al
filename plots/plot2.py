import matplotlib.pyplot as plt
import numpy as np

# Data from the image
categories = [
    "Absolutely nothing: I have let it go",
    "I have deleted the Hate Speech message, and unfriended/blocked the person who has attacked me",
    "I have fought back, and tried to engage with the person who has attacked me",
    "I have reported this to the social media provider",
    "I did not know what to do, and I have sought advice from someone else",
    "I have reported this to the police",
    "Other"
]
male_values = [49, 31, 11, 10, 6, 2, 0]
female_values = [32, 32, 27, 13, 12, 4, 0]

# Define bar positions
y = np.arange(len(categories))

# Bar width
bar_width = 0.35

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 6))

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

# Add category text to the right side
for i, category in enumerate(categories):
    ax.text(105, i, category, fontsize=10, verticalalignment='center', color='black')

# Adjust the xlim to accommodate the text
ax.set_xlim(0, 130)

# Save the plot to a file
plt.tight_layout()
plt.savefig('plots/reproduced_graph_with_text.png', format='png')

# Display the plot
plt.show()

