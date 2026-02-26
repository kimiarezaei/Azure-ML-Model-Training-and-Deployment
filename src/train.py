import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split  
import random
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
import logging

from utils import load_config, setup_logger


def train(config_path: str = None, data_dir: str = None, output_dir: str = None):
    # Paths
    config_path = Path(config_path or Path(__file__).parent.parent / "config/config.yml")
    config = load_config(config_path)
    data_dir = Path(config["data"]["data_path"])
    output_dir = Path(output_dir or Path(__file__).parent.parent / config["dirs"]["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    # set up logger
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Start training...")

    # Set random seeds
    np.random.seed(config['training']['random_state'])
    random.seed(config['training']['random_state'])

    # Read data
    data = pd.read_csv(data_dir)
    # Handle missing data
    data = data.drop(columns=['Name', 'Ticket', 'Cabin'])
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)   # Encode categorical variables

    # Train and test data
    X = data.drop(columns=['Survived'])
    y = data['Survived']
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=config['training']['test_size'], random_state=config['training']['random_state'], stratify=y)

    # Train a model
    model = CatBoostClassifier(
    iterations=config['training']['iterations'],
    learning_rate=config['training']['learning_rate'],
    depth=config['training']['depth'],
    random_seed=config['training']['random_state'],
    verbose=False
)
    model.fit(X_train, y_train)

    # Model prediction
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    model_path = output_dir / "model.cbm"
    model.save_model(str(model_path))

    logger.info(f"Model saved")
    logger.info(f"Model accuracy: {accuracy:.4f}")

    # save model as onnx
    onnx_path = output_dir / "model.onnx"
    model.save_model(str(onnx_path), format="onnx")
 
    logger.info(f"Model saved as ONNX format")

    # Register model for azure ML
    ml_client = MLClient.from_config()

    model = Model(
        name="CatBoost-model",
        path="outputs", 

        type="custom_model"
    )

    ml_client.models.create_or_update(model)





if __name__ == "__main__":
    train()