import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris flower dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the machine learning model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Check model accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model accuracy:", accuracy)

# Save the trained model and flower names
saved_data = {
    "model": model,
    "target_names": iris.target_names
}

joblib.dump(saved_data, "model.pkl")

print("Model trained and saved successfully!")
