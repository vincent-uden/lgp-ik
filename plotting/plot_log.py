import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
csv_file = '/media/hdd/github/lgp-ik/log.txt'  # Replace with your CSV file's path
df = pd.read_csv(csv_file)

# Extract data from the DataFrame
generation = df.index  # Assuming the index represents generation number
best_fitness = df['best_fitness']
mean_fitness = df['mean_fitness']
median_fitness = df['median_fitness']

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(generation, best_fitness, label='Best Fitness')
plt.plot(generation, mean_fitness, label='Mean Fitness')
plt.plot(generation, median_fitness, label='Median Fitness')

# Customize the plot
plt.title('Fitness Time Series')
plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()

