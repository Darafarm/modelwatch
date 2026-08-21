import pandas as pd

df = pd.read_json("orders.json")

print("Shape (rows, columns):", df.shape)
print("\nColumn names and data types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

print("\nUnique categories:")
print(df["product"].unique())
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())