import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    return df

def handle_outliers(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64'])

    Q1 = numerical_columns.quantile(0.25)
    Q3 = numerical_columns.quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1['sepal.width'] - 1.5 * IQR['sepal.width']
    upper_limit = Q3['sepal.width'] + 1.5 * IQR['sepal.width']

    df_cleaned = df[(df['sepal.width'] >= lower_limit) & (df['sepal.width'] <= upper_limit)]
    return df_cleaned

def normalize_data(df_cleaned):
    variety = df_cleaned['variety']
    numerical_columns_cleaned = df_cleaned.drop(columns=['variety'])

    scaler = StandardScaler()
    numerical_columns_normalized = pd.DataFrame(scaler.fit_transform(numerical_columns_cleaned), columns=numerical_columns_cleaned.columns)

    df_normalized = numerical_columns_normalized.copy()
    df_normalized['variety'] = variety.reset_index(drop=True)
    return df_normalized

def preprocess_iris_data(file_path, output_file_path):
    df = load_and_clean_data(file_path)
    df_cleaned = handle_outliers(df) 
    df_normalized = normalize_data(df_cleaned) 

    df_normalized.to_csv(output_file_path, index=False)
    return df_normalized

if __name__ == "__main__":
    input_file = 'data/iris.csv'
    output_file = 'data/processed_iris.csv'
    df_prepared = preprocess_iris_data(input_file, output_file)