import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['variety'])
    y = df['variety']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_file}")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    evaluation_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    return evaluation_metrics

def save_eval_metrics(metrics, output_file):
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to {output_file}")

if __name__ == "__main__":
    input_file = 'data/processed_iris.csv'
    model_file = 'models/decision_tree_model.pkl'
    output_file = 'results/evaluation_metrics.json'
    
    X, y = load_data(input_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = load_model(model_file)
    evaluation_metrics = evaluate_model(model, X_test, y_test)
    save_eval_metrics(evaluation_metrics, output_file)
    print(evaluation_metrics)