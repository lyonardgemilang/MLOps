from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import to_onnx
import onnxruntime as rt
import pickle
import joblib
import time
import os
import pandas as pd

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Directories for saving models
os.makedirs("models", exist_ok=True)

results = []

# 1. Pickle
start_time = time.time()
with open("models/model_pickle.pkl", "wb") as f:
    pickle.dump(model, f)
pickle_save_time = time.time() - start_time

start_time = time.time()
with open("models/model_pickle.pkl", "rb") as f:
    loaded_pickle_model = pickle.load(f)
pickle_load_time = time.time() - start_time

pickle_size = os.path.getsize("models/model_pickle.pkl")

start_time = time.time()
joblib.dump(model, "models/model_joblib.pkl")
joblib_save_time = time.time() - start_time

start_time = time.time()
loaded_joblib_model = joblib.load("models/model_joblib.pkl")
joblib_load_time = time.time() - start_time

joblib_size = os.path.getsize("models/model_joblib.pkl")

results.append({
    "Method": "Pickle",
    "Save Time (s)": pickle_save_time,
    "Load Time (s)": pickle_load_time,
    "File Size (bytes)": pickle_size
})

results.append({
    "Method": "Joblib",
    "Save Time (s)": joblib_save_time,
    "Load Time (s)": joblib_load_time,
    "File Size (bytes)": joblib_size
})

# Display results
results_df = pd.DataFrame(results)

# onx = to_onnx(clr, X[:1])
# with open("rf_iris.onnx", "wb") as f:
#     f.write(onx.SerializeToString())