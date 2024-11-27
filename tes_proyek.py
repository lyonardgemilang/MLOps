from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Load dataset
data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment once outside of the run
mlflow.set_experiment('Default')

def train_model(alpha):
    # Start a new MLflow run
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        # Log parameters and metrics
        mlflow.log_param('alpha', alpha)
        mlflow.log_metric('mse', mse)
        mlflow.sklearn.log_model(model, 'model')

        print(f'Model trained with alpha={alpha}, MSE={mse}')

# Train the model with alpha value
train_model(alpha=0.1)