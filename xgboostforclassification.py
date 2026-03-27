# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =========================
# 2. Load Dataset
# =========================
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# =========================
# 3. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Feature Scaling (optional but good practice)
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 5. Model Initialization
# =========================
model = XGBClassifier(
    n_estimators=200,        # number of trees
    learning_rate=0.05,      # slower learning = better generalization
    max_depth=4,             # tree depth
    subsample=0.8,           # row sampling
    colsample_bytree=0.8,    # feature sampling
    random_state=42,
    eval_metric='logloss'
)

# =========================
# 6. Train Model
# =========================
model.fit(X_train, y_train)

# =========================
# 7. Predictions
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# 8. Evaluation
# =========================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 9. Feature Importance
# =========================
importance = model.feature_importances_

plt.figure()
plt.bar(range(len(importance)), importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# =========================
# 10. Predict New Data
# =========================
# Example: take first test sample
new_sample = X_test[0].reshape(1, -1)

prediction = model.predict(new_sample)
probability = model.predict_proba(new_sample)[:, 1]

print("\nNew Sample Prediction:", prediction[0])
print("Probability of Class 1:", probability[0])