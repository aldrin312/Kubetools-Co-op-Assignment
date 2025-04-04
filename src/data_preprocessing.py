import kfp
from kfp import dsl
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib



@kfp.dsl.component(base_image='python:3.10')
def preprocess_data(
    output_X_scaled_csv: kfp.dsl.OutputPath(str), # type: ignore
    output_scaler: kfp.dsl.OutputPath(str), # type: ignore
    output_y: kfp.dsl.OutputPath(list) # type: ignore
):
    df1 = pd.read_csv('./models/diabetes.csv')  
    df2 = pd.read_csv('./models/diabetes2.csv')  
    
    # Combine the datasets
    df = pd.concat([df1, df2], ignore_index=True)

    # Drop target column and normalize the features
    X = df.drop(columns=['Diabetic'])
    y = df['Diabetic']
    
    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for later use
    joblib.dump(scaler, '/tmp/scaler.pkl')
    output_scaler.set('/tmp/scaler.pkl')
    
    # Save scaled data
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df.to_csv('/tmp/X_scaled.csv', index=False)
    output_X_scaled_csv.set('/tmp/X_scaled.csv')
    
    # Set the target variable output
    output_y.set(y.to_list())