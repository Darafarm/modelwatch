import pandas as pd

df = pd.read_json("orders.json")

print("Average price per product:")
print(df.groupby("product")["price"].mean().sort_values())

print("\nCorrelation between quantity and price:")
print(df["quantity"].corr(df["price"]))