import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split  
import random
import argparse
import os
import joblib
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model


# Paths
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', 
                    default=r'./data/Titanic-Dataset.csv', 
                    help='dataset')

parser.add_argument('--save_dir', 
                    default=r'./outputs',
                    help='Directory for saving the result')

args = parser.parse_args()

# Set random seeds
np.random.seed(42)
random.seed(42)

# Read data
data = pd.read_csv(args.data_dir)
# Handle missing data
data = data.drop(columns=['Name', 'Ticket', 'Cabin'])
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)   # Encode categorical variables

# Train and test data
X = data.drop(columns=['Survived'])
y = data['Survived']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model
model_path = os.path.join(args.save_dir, "model.pkl")
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")
print(f"Test Accuracy: {accuracy*100:.2f}")

# Register model for azure ML
ml_client = MLClient.from_config()

model = Model(
    name="RF-model",
    path=args.save_dir,   
    type="custom_model"
)

ml_client.models.create_or_update(model)

print("Model registered successfully")