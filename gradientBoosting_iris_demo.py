import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

class GradientBoostingRegressorManual:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.initial_prediction = None

    def fit(self, X, y):
        # Step 1: Initial prediction (mean of target)
        self.initial_prediction = np.mean(y)
        y_pred = np.full(shape=y.shape, fill_value=self.initial_prediction)

        # Step 2: Iteratively train trees
        for i in range(self.n_estimators):
            # Compute residuals
            residuals = y - y_pred

            # Train weak learner
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Predict residuals
            pred = tree.predict(X)

            # Update prediction
            y_pred += self.learning_rate * pred

            # Save model
            self.models.append(tree)

    def predict(self, X):
        # Start with initial prediction
        y_pred = np.full(shape=(X.shape[0],), fill_value=self.initial_prediction)

        # Add contributions from each tree
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)

        return y_pred

# Load Iris dataset for regression example: predict petal width from other features
iris = load_iris()
X = iris.data[:, [0, 1, 2]]  # sepal length, sepal width, petal length
y = iris.data[:, 3]  # petal width (continuous target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train manual implementation
manual_gb = GradientBoostingRegressorManual(n_estimators=100, learning_rate=0.1, max_depth=3)
manual_gb.fit(X_train, y_train)
manual_pred = manual_gb.predict(X_test)

# Evaluate manual model
manual_mse = mean_squared_error(y_test, manual_pred)
manual_r2 = r2_score(y_test, manual_pred)

print("Manual Gradient Boosting Results:")
print(f"MSE: {manual_mse:.4f}")
print(f"R²: {manual_r2:.4f}")

# Compare with sklearn implementation
sklearn_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
sklearn_gb.fit(X_train, y_train)
sklearn_pred = sklearn_gb.predict(X_test)

sklearn_mse = mean_squared_error(y_test, sklearn_pred)
sklearn_r2 = r2_score(y_test, sklearn_pred)

print("\nSklearn Gradient Boosting Results:")
print(f"MSE: {sklearn_mse:.4f}")
print(f"R²: {sklearn_r2:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, manual_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Petal Width')
plt.ylabel('Predicted Petal Width (Manual)')
plt.title('Manual GB: Predictions vs Actual')

plt.subplot(1, 2, 2)
plt.scatter(y_test, sklearn_pred, alpha=0.7, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Petal Width')
plt.ylabel('Predicted Petal Width (Sklearn)')
plt.title('Sklearn GB: Predictions vs Actual')
plt.tight_layout()
plt.show()

