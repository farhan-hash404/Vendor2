import numpy as np
from sklearn.tree import DecisionTreeRegressor

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