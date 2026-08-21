import pandas as pd
from pymongo import MongoClient

df = pd.read_json("synthetic_orders.json")

client = MongoClient("mongodb://localhost:27017/")
db = client["modelwatch_db"]
synthetic_orders_collection = db["synthetic_orders"]

synthetic_orders_collection.delete_many({})

records = df.to_dict("records")
synthetic_orders_collection.insert_many(records)

print(f"Inserted {synthetic_orders_collection.count_documents({})} synthetic orders")