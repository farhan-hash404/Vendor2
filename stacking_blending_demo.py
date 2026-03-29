import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load Iris dataset (classification)
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Train-test split (67%-33%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
print(f"Classes: {np.unique(y)}")

# === Stacking Classifier ===
print("\n=== Stacking Ensemble ===")
# Level-0 estimators
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]

# Stacking with LogisticRegression meta-learner
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

# Train and evaluate
stacking_model.fit(X_train, y_train)
stack_pred = stacking_model.predict(X_test)
stack_acc = accuracy_score(y_test, stack_pred)

print(f"Stacking Accuracy: {stack_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, stack_pred, target_names=iris.target_names))

# Cross-validation score
stack_cv = cross_val_score(stacking_model, X_train, y_train, cv=5)
print(f"Stacking CV Score (mean): {stack_cv.mean():.4f} (+/- {stack_cv.std() * 2:.4f})")

# === Blending (Manual Implementation) ===
print("\n=== Blending Ensemble ===")
# Further split train into train/val for blending
X_train_blend, X_val, y_train_blend, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

# Train base models on train_blend
rf_blend = RandomForestClassifier(n_estimators=100, random_state=42)
svc_blend = SVC(probability=True, random_state=42)
lr_blend = LogisticRegression(random_state=42)

rf_blend.fit(X_train_blend, y_train_blend)
svc_blend.fit(X_train_blend, y_train_blend)
lr_blend.fit(X_train_blend, y_train_blend)

# Blend predictions on val (probabilities)
rf_val_proba = rf_blend.predict_proba(X_val)
svc_val_proba = svc_blend.predict_proba(X_val)
lr_val_proba = lr_blend.predict_proba(X_val)

# Stack val predictions as meta-features
val_meta = np.hstack([rf_val_proba, svc_val_proba, lr_val_proba])
meta_model = LogisticRegression(random_state=42)
meta_model.fit(val_meta, y_val)

# Final predictions on test
rf_test_proba = rf_blend.predict_proba(X_test)
svc_test_proba = svc_blend.predict_proba(X_test)
lr_test_proba = lr_blend.predict_proba(X_test)
test_meta = np.hstack([rf_test_proba, svc_test_proba, lr_test_proba])
blend_pred = meta_model.predict(test_meta)
blend_acc = accuracy_score(y_test, blend_pred)

print(f"Blending Accuracy: {blend_acc:.4f}")

# Compare base models
base_accs = {}
for name, model in [('RF', rf_blend), ('SVC', svc_blend), ('LR', lr_blend)]:
    base_pred = model.predict(X_test)
    base_acc = accuracy_score(y_test, base_pred)
    base_accs[name] = base_acc
    print(f"{name} Base Accuracy: {base_acc:.4f}")

# === Plots ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Accuracy comparison bar plot
models = ['Stacking', 'Blending'] + list(base_accs.keys())
accs = [stack_acc, blend_acc] + list(base_accs.values())
axes[0,0].bar(models, accs, color=['blue', 'green', 'orange', 'red', 'purple'])
axes[0,0].set_title('Model Accuracies')
axes[0,0].set_ylabel('Accuracy')
plt.setp(axes[0,0].get_xticklabels(), rotation=45)

# Confusion Matrix for Stacking
cm = confusion_matrix(y_test, stack_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[0,1].set_title('Stacking Confusion Matrix')

# Confusion Matrix for Blending
cm_blend = confusion_matrix(y_test, blend_pred)
sns.heatmap(cm_blend, annot=True, fmt='d', cmap='Greens', ax=axes[1,0],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[1,0].set_title('Blending Confusion Matrix')

# Feature Importances (from RF in stacking)
if hasattr(stacking_model.named_estimators_['rf'], 'feature_importances_'):
    importances = stacking_model.named_estimators_['rf'].feature_importances_
    indices = np.argsort(importances)[::-1]
    axes[1,1].bar(range(X.shape[1]), importances[indices])
    axes[1,1].set_title('RF Feature Importances (Stacking)')
    axes[1,1].set_xticks(range(X.shape[1]))
    axes[1,1].set_xticklabels([feature_names[i][:-5] for i in indices], rotation=45)

plt.tight_layout()
plt.show()

print("\nDemo complete! Check plots and results above.")
