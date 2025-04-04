import kfp
from kfp import dsl
import optuna
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split

def hyperparameter_tuning_component(X_scaled_csv: str, y: list, best_model_output: str):
    
    X_scaled = pd.read_csv(X_scaled_csv)
    y = pd.Series(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Optuna objective function
    def objective(trial):
        learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
        max_depth = trial.suggest_int('max_depth', 3, 10)

        # Initialize the model with hyperparameters
        model = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

        return roc_auc

    # Set up Optuna study and optimize the objective
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    # Save the best model
    best_params = study.best_params
    best_model = GradientBoostingClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, best_model_output)

    return best_params, study.best_value

@kfp.dsl.component(base_image='python:3.10')
def hyperparameter_tuning(X_scaled_csv: str, y: list, best_model_output: str):
    hyperparameter_tuning_component(X_scaled_csv, y, best_model_output)
