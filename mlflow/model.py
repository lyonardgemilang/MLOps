from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Set tracking URI (choose one based on your setup)
mlflow.set_tracking_uri('http://127.0.0.1:5000')  # if using MLflow server
print("Tracking URI:", mlflow.get_tracking_uri())

# Set experiment once outside of the run
mlflow.set_experiment('Default')

# def train_model(alpha):
#     # Start a new MLflow run
#     with mlflow.start_run():
#         model = LinearRegression()
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         mse = mean_squared_error(y_test, preds)

#         # Log parameters and metrics
#         mlflow.log_param('alpha', alpha)
#         mlflow.log_metric('mse', mse)
#         mlflow.sklearn.log_model(model, 'model')

#         print(f'Model trained with alpha={alpha}, MSE={mse}')

# # Train the model with alpha value
# train_model(alpha=0.1)
# train_model(alpha=0.5)

model_uri = "runs:/28e92b9577794a738f76b611b362b29a/model"
model = mlflow.sklearn.load_model(model_uri)

new_predictions = model.predict(X_test)

residuals = y_test - new_predictions
plt.hist(residuals, bins=20)
plt.savefig('residuals_hist.png')