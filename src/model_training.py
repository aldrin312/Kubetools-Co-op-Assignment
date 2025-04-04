import kfp
from kfp import dsl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

def train_and_evaluate_component(X_scaled_csv: str, y: list, model_output: str):
    
    X_scaled = pd.read_csv(X_scaled_csv)
    y = pd.Series(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train GradientBoosting model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Save the model
    joblib.dump(model, model_output)
    return accuracy, roc_auc

# Define the component
@kfp.dsl.component(base_image='python:3.10')
def train_and_evaluate(X_scaled_csv: str, y: list, model_output: str):
    train_and_evaluate_component(X_scaled_csv, y, model_output)
