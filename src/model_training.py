from sklearn.model_selection import train_test_split
from utils import get_preprocessed_data, train_and_evaluate_model, save_model

# Load the preprocessed data
X_scaled, y = get_preprocessed_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train and evaluate the model
accuracy, roc_auc, model = train_and_evaluate_model(X_train, y_train, X_test, y_test)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Save the trained model
save_model(model)
