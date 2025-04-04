import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load both diabetes datasets
df1 = pd.read_csv('./datasets/diabetes.csv')
df2 = pd.read_csv('./datasets/diabetes2.csv')

df = pd.concat([df1, df2], ignore_index=True)

print(df.head())

# Preprocessing function
def preprocess_data(df):
    X = df.drop(columns=['Diabetic'])  
    y = df['Diabetic']  
    
    # Normalize feature data using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Preprocess the data
X_train, y_train = preprocess_data(df)

# Check the shapes of the resulting data
print(f"Feature data shape: {X_train.shape}")
print(f"Target data shape: {y_train.shape}")
