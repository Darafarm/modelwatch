# ModelWatch

ModelWatch is a machine learning model serving API built with FastAPI and MongoDB. It predicts an order's price and, instead of returning a single guess, gives a genuine statistical range for how confident that prediction actually is. This project extends the infrastructure work from [MongoGuard](https://github.com/Darafarm/mongoguard), applying its MongoDB benchmarking knowledge to a real, live serving workload.

## What this actually does

The API takes an order ID, looks up that order's product, quantity, customer type, and region from MongoDB, and returns a predicted price along with a 90 percent prediction interval. The interval is computed properly using statsmodels rather than faked, since a plain linear regression has no built in confidence measure the way a classifier does.

Along the way this project uncovered that its original MongoGuard order data had no real relationship between product, quantity, and price, since that dataset was generated purely for database benchmarking. A synthetic dataset with a real, known pricing formula was built instead, giving the model something genuine to learn from.

## Project structure

`main.py` is the FastAPI application itself, with a health check endpoint and the prediction endpoint.

`generate_synthetic_orders.py` builds the training dataset from a defined pricing formula involving product, quantity, customer discount, and region.

`load_orders_to_mongo.py` loads that generated dataset into a local MongoDB instance.

`train_model_log.py` trains the regression pipeline, evaluates it honestly on a held out test set, computes prediction intervals, and saves both the scikit learn pipeline and the statsmodels model to disk.

`explore_data.py` was used to inspect the original MongoGuard dataset and confirm it lacked a usable price relationship.

## Running it locally

You will need a local MongoDB instance. The simplest way is Docker.
