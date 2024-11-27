# Import necessary libraries
import tensorflow_decision_forests as tfdf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert to DataFrame for compatibility with TensorFlow Decision Forests
data = pd.DataFrame(X, columns=iris.feature_names)
data['label'] = y

# Split dataset using Scikit-Learn's train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert the split data into TensorFlow datasets
train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, label="label")
test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, label="label")

# Initialize and train the Random Forest model
model = tfdf.keras.RandomForestModel()
model.fit(train_dataset)

# Evaluate the model
evaluation = model.evaluate(test_dataset)

# Output the evaluation results
print("Evaluation Results:", evaluation)

# Optionally, display feature importance
print("Feature Importances:")
for feature in model.make_inspector().variable_importances().items():
    print(feature)
