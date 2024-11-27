import pandas as pd
import onnx
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.values.ravel())

# Test accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Serialization to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(rf_model, initial_types=initial_type)
with open("random_forest.onnx", "wb") as file:
    file.write(onnx_model.SerializeToString())

# Print results
print(f"Model Accuracy: {accuracy}")
print("Serialization completed using ONNX.")

# Print input details
print("Inputs:")
for input in onnx_model.graph.input:
    print(input.name, input.type)

# Print output details
print("\nOutputs:")
for output in onnx_model.graph.output:
    print(output.name, output.type)
