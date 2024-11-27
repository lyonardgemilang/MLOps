import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['variety'])
    y = df['variety']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded

def train_model(X_train, y_train):
    param_grid = {'max_depth': range(1, 11)}
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_max_depth = grid_search.best_params_['max_depth']
    best_accuracy = grid_search.best_score_
    print(f"Best max_depth: {best_max_depth}, Best cross-validated accuracy: {best_accuracy}")
    model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {output_file}")

if __name__ == "__main__":
    input_file = 'data/processed_iris.csv'
    model_output_file = 'models/decision_tree_model.pkl'
    X, y = load_data(input_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    save_model(model, model_output_file)