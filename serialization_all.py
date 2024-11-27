import pandas as pd
import onnx
import pickle
import joblib
import tensorflow as tf
import time
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder, StandardScaler

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train.values.ravel())

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

tf_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_tf.shape[1],)),
    Dense(16, activation='relu'),
    Dense(y_train_tf.shape[1], activation='softmax')
])

tf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train_tf, y_train_tf, epochs=20, batch_size=8, verbose=0)

y_pred_tf = tf_model.predict(X_test_tf)
y_pred_tf_classes = tf.argmax(y_pred_tf, axis=1).numpy()
y_test_tf_classes = tf.argmax(y_test_tf, axis=1).numpy()
tf_accuracy = accuracy_score(y_test_tf_classes, y_pred_tf_classes)
print(f"TensorFlow Model Accuracy: {tf_accuracy}")

serialization_results = []

pickle_file = "random_forest_pickle.pkl"
start_save = time.time()
with open(pickle_file, "wb") as file:
    pickle.dump(rf_model, file)
end_save = time.time()
pickle_save_time = end_save - start_save

start_load = time.time()
with open(pickle_file, "rb") as file:
    pickle_model = pickle.load(file)
end_load = time.time()
pickle_load_time = end_load - start_load

pickle_file_size = os.path.getsize(pickle_file) / 1024
serialization_results.append(("Pickle", pickle_save_time, pickle_load_time, pickle_file_size))

joblib_file = "random_forest_joblib.pkl"
start_save = time.time()
joblib.dump(rf_model, joblib_file)
end_save = time.time()
joblib_save_time = end_save - start_save

start_load = time.time()
joblib_model = joblib.load(joblib_file)
end_load = time.time()
joblib_load_time = end_load - start_load

joblib_file_size = os.path.getsize(joblib_file) / 1024
serialization_results.append(("Joblib", joblib_save_time, joblib_load_time, joblib_file_size))

onnx_file = "random_forest.onnx"
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(rf_model, initial_types=initial_type)

start_save = time.time()
with open(onnx_file, "wb") as file:
    file.write(onnx_model.SerializeToString())
end_save = time.time()
onnx_save_time = end_save - start_save

start_load = time.time()
onnx_loaded_model = onnx.load(onnx_file)
end_load = time.time()
onnx_load_time = end_load - start_load

onnx_file_size = os.path.getsize(onnx_file) / 1024
serialization_results.append(("ONNX", onnx_save_time, onnx_load_time, onnx_file_size))

h5_file = "tensorflow_model.h5"
start_save = time.time()
tf_model.save(h5_file, save_format='h5')
end_save = time.time()
h5_save_time = end_save - start_save

start_load = time.time()
loaded_h5_model = tf.keras.models.load_model(h5_file)
end_load = time.time()
h5_load_time = end_load - start_load

h5_file_size = os.path.getsize(h5_file) / 1024
serialization_results.append(("HDF5", h5_save_time, h5_load_time, h5_file_size))

# tf_dir = "tensorflow_model_tf"
# start_save = time.time()
# tf_model.save(tf_dir, save_format='tf')
# end_save = time.time()
# tf_save_time = end_save - start_save

# start_load = time.time()
# loaded_tf_model = tf.keras.models.load_model(tf_dir)
# end_load = time.time()
# tf_load_time = end_load - start_load

# tf_dir_size = sum(os.path.getsize(os.path.join(dirpath, file)) for dirpath, _, filenames in os.walk(tf_dir) for file in filenames) / 1024
# serialization_results.append(("TensorFlow SavedModel", tf_save_time, tf_load_time, tf_dir_size))

print("\nSerialization Results:")
print(f"{'Method':<25}{'Save Time (s)':<15}{'Load Time (s)':<15}{'File Size (KB)':<15}")
for method, save_time, load_time, file_size in serialization_results:
    print(f"{method:<25}{save_time:<15.6f}{load_time:<15.6f}{file_size:<15.2f}")
    
loaded_h5_model = tf.keras.models.load_model(h5_file)