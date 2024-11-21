'''
import tensorflow as tf
import numpy as np
import os

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Base directory of the script
MODEL_DIR = os.path.join(BASE_DIR, "models")           # Directory to save models
os.makedirs(MODEL_DIR, exist_ok=True)                  # Create models directory if it doesn't exist
MODEL_PATH = os.path.join(MODEL_DIR, "regression_model.h5")  # Path to save the model

# Generate dummy data for training
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
y = np.sum(X, axis=1)         # Simple summation as the target

# Define a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)  # Single output
])

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# Train the model
print("Training the model...")
history = model.fit(X, y, epochs=10, batch_size=32)

# Save the trained model
print(f"Saving the model to: {MODEL_PATH}")
model.save(MODEL_PATH)

print("Model training and saving completed successfully!")

# Save the trained model

'''
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression

# Generate synthetic data (for regression)
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_class, y_class = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Split the data into training and test sets for both models
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Create and train a regression model
regression_model = LinearRegression()
regression_model.fit(X_train_reg, y_train_reg)

# Save the trained regression model
regression_model_path = "models/regression_model.pkl"
joblib.dump(regression_model, regression_model_path)

print(f"Regression model saved to {regression_model_path}")

# Create and train a classification model
classification_model = LogisticRegression()
classification_model.fit(X_train_class, y_train_class)

# Save the trained classification model
classification_model_path = "models/classification_model.pkl"
joblib.dump(classification_model, classification_model_path)

print(f"Classification model saved to {classification_model_path}")

