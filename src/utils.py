import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from data_pipeline import load_and_preprocess_data
import os

# Function to load and preprocess data
def get_preprocessed_data():
    X_scaled, y = load_and_preprocess_data()
    return X_scaled, y

# Function to train and evaluate the model
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model=None):
    if model is None:
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] 

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    return accuracy, roc_auc, model

# Function to save the trained model
def save_model(model, filename='diabetes_model.pkl'):
    file_path = os.path.join('models', filename)
    joblib.dump(model, file_path)
    print(f"Model saved in '{file_path}'")

# Function to load the trained model
def load_model(filename='./models/diabetes_model.pkl'):
    return joblib.load(filename)
