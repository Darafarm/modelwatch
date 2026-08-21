from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["modelwatch_db"]
collection = db["customers"]

collection.insert_one({"customer_id": "C1042", "category": "electronics", "quantity": 3})
collection.insert_one({"customer_id": "C1043", "category": "Furniture", "quantity": 5})
result = collection.find_one({"customer_id": "C1043"})
print(result)