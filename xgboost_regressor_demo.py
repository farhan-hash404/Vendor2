import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Note: Install xgboost first if needed:
# pip install xgboost

print("Loading California Housing dataset...")
housing = fetch_california_housing()
X = housing.data  # Features: MedInc, HouseAge, AveRooms, etc.
y = housing.target  # Median house value in $100,000s

print(f"Dataset shape: {X.shape}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining XGBoost Regressor...")
# XGBoost regressor with parameters similar to gradient boosting demos
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    verbosity=0  # Reduce output noise
)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nXGBoost Results on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot: Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual House Value ($100k)')
plt.ylabel('Predicted House Value ($100k)')
plt.title('XGBoost: Actual vs Predicted')

# Feature Importance
plt.subplot(1, 2, 2)
feature_importance = xgb_model.feature_importances_
top_features = np.argsort(feature_importance)[-5:]  # Top 5
plt.barh(range(len(top_features)), feature_importance[top_features])
plt.yticks(range(len(top_features)), [housing.feature_names[i] for i in top_features])
plt.xlabel('Feature Importance')
plt.title('Top 5 Feature Importances')
plt.tight_layout()
plt.show()

# Example prediction on new data
print("\nExample prediction:")
new_house = np.array([[8.32, 41.0, 6.98, 18.5, 49.0, 3.17, 2.8, 0.52]])  # Sample feature values
predicted_value = xgb_model.predict(new_house)[0]
print(f"Predicted house value for new sample: ${predicted_value*100000:.0f}")

print("\nXGBoost regressor demo complete!")
