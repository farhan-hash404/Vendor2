# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load dataset
# Example dataset file
df = pd.read_csv("vendor_invoices.csv")

# Example columns in dataset
# Vendor, Distance, Weight, Shipment_Type, Freight_Cost

# Step 2: Convert categorical data into numbers
df = pd.get_dummies(df, columns=["Vendor", "Shipment_Type"])

# Step 3: Select features (input variables)
X = df.drop("Freight_Cost", axis=1)

# Target variable (what we want to predict)
y = df["Freight_Cost"]

# Step 4: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train model
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)

# Step 6: Make predictions
predictions = model.predict(X_test)

# Step 7: Evaluate model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Step 8: Predict new vendor invoice
new_data = pd.DataFrame({
    "Distance": [200],
    "Weight": [50],
})

# If categorical columns exist, they must match training columns
new_data = new_data.reindex(columns=X.columns, fill_value=0)

predicted_cost = model.predict(new_data)

print("Predicted Vendor Invoice Cost:", predicted_cost[0])