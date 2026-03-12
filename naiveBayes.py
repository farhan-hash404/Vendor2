# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

# Example dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 48],
    'Income': [15000, 29000, 48000, 51000, 50000, 60000, 52000],
    'Buy': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[['Age', 'Income']]
y = df['Buy']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Predicted:", y_pred)
print("Accuracy:", accuracy)