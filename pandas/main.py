import pandas as pd

# Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# Reading data from a CSV
df = pd.read_csv('data.csv')

# Data selection
df['Name']  # Select a column
df.loc[0]   # Select a row by index

# Data cleaning
df.dropna()  # Remove rows with missing values
df.fillna(0) # Replace NaNs with 0

# Grouping and aggregation
grouped = df.groupby('Category').sum()

# Exporting data to CSV
df.to_csv('output.csv', index=False)
