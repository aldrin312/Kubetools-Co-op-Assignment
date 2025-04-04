import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load both diabetes datasets
def load_and_preprocess_data():
    df1 = pd.read_csv('./datasets/diabetes.csv')
    df2 = pd.read_csv('./datasets/diabetes2.csv')

    # Concatenate both datasets
    df = pd.concat([df1, df2], ignore_index=True)

    # Preprocessing function
    def preprocess_data(df):
        X = df.drop(columns=['Diabetic'])
        y = df['Diabetic']

        # Normalize data using MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y

    # Preprocess the data
    X_scaled, y = preprocess_data(df)

    return X_scaled, y

if __name__ == '__main__':
    X_scaled, y = load_and_preprocess_data()
    print(f"Data preprocessing complete: {X_scaled.shape} features, {y.shape} target")
