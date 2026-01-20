import json
import joblib
import os


# this file handles requests and returns predictions

# Load model once
def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)

# Process every request
def run(raw_data):
    data = json.loads(raw_data)
    preds = model.predict(data["data"])
    return preds.tolist()
