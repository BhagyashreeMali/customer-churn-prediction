from src.data_preprocessing import load_and_preprocess_data
from src.model_training import (
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting
)
from src.evaluation import evaluate_model
import joblib

# Load and preprocess data
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(
    "data/churn.csv"
)

# Train models
lr_model = train_logistic_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)
gb_model = train_gradient_boosting(X_train, y_train)

# Evaluate models
evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
evaluate_model(rf_model, X_test, y_test, "Random Forest")
evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")

# Save best model
joblib.dump(rf_model, "customer_churn_model.pkl")
print("\n✅ Model saved as customer_churn_model.pkl")