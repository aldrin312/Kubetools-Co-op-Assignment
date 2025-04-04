import optuna
from utils import get_preprocessed_data, train_and_evaluate_model, load_model

# Load preprocessed data
X_scaled, y = get_preprocessed_data()

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Load the saved model
clf = load_model('./models/diabetes_model.pkl')

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameter search space
    learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
    max_depth = trial.suggest_int('max_depth', 3, 10)

    # Fine-tune the GradientBoostingClassifier with suggested hyperparameters
    clf.set_params(learning_rate=learning_rate, max_depth=max_depth)

    # Train and evaluate the model
    accuracy, roc_auc, _ = train_and_evaluate_model(X_train, y_train, X_test, y_test, model=clf)

    # Log the metrics and return the ROC AUC score for Optuna optimization
    print(f"Trial results: learning_rate={learning_rate}, max_depth={max_depth}, Accuracy={accuracy:.4f}, ROC AUC={roc_auc:.4f}")
    
    return roc_auc

# Set up the Optuna study to optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Output the best hyperparameters and corresponding metrics
print("Best Hyperparameters:")
print(study.best_params)
print(f"Best ROC AUC: {study.best_value:.4f}")
