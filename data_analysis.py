import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Explore the structure of the dataset (data types, missing values)
print("\nData types and missing values:")
print(df.info())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Clean the dataset (if there were missing values, you could fill them here)
# df.fillna(method='ffill', inplace=True)  # Example of filling missing values

# Task 2: Basic Data Analysis
# Compute basic statistics for numerical columns
print("\nBasic statistics for numerical columns:")
print(df.describe())

# Group by the 'species' column and compute the mean
grouped_by_species = df.groupby('species').mean()
print("\nAverage values by species:")
print(grouped_by_species)

# Task 3: Data Visualization
# Line chart: Sepal Length by Species
plt.figure(figsize=(10, 6))
sns.lineplot(x='species', y='sepal length (cm)', data=df, marker='o')
plt.title("Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length (cm)")
plt.show()

# Bar chart: Average Petal Length by Species
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# Histogram: Distribution of Sepal Width
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='blue')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot: Sepal Length vs Petal Length
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()

# Observations/Findings
# - The petal length varies significantly between species, with Iris-virginica having the longest petals.
# - Sepal length and petal length seem to be positively correlated.
