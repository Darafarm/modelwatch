import pandas as pd
import numpy as np

np.random.seed(42)

n_orders = 10000

products = ["Phone", "Monitor", "Headphones", "Laptop", "SSD",
            "USB Hub", "Mouse", "Webcam", "Keyboard", "Tablet"]

unit_price = {
    "Phone": 650, "Monitor": 220, "Headphones": 90, "Laptop": 950,
    "SSD": 110, "USB Hub": 25, "Mouse": 20, "Webcam": 45,
    "Keyboard": 55, "Tablet": 380
}

regions = ["North America", "Europe", "Asia"]
region_multiplier = {
    "North America": 1.00,
    "Europe": 1.08,
    "Asia": 0.95
}

order_id = np.arange(1, n_orders + 1)
product = np.random.choice(products, size=n_orders)
quantity = np.random.randint(1, 11, size=n_orders)
customer_type = np.random.choice(["new", "returning"], size=n_orders, p=[0.6, 0.4])
region = np.random.choice(regions, size=n_orders, p=[0.5, 0.3, 0.2])

base_price = np.array([unit_price[p] for p in product]) * quantity
discount_multiplier = np.where(customer_type == "returning", 0.90, 1.0)
region_mult = np.array([region_multiplier[r] for r in region])
noise = np.random.normal(loc=0, scale=25, size=n_orders)

price = base_price * discount_multiplier * region_mult + noise
price = np.maximum(price, 1.0)   # real prices can never be $0 or negative
price = np.round(price, 2)

df = pd.DataFrame({
    "order_id": order_id,
    "product": product,
    "quantity": quantity,
    "customer_type": customer_type,
    "region": region,
    "price": price
})

df.to_json("synthetic_orders.json", orient="records")
print(df.head())
print("\nShape:", df.shape)
print("\nMinimum price:", df["price"].min())