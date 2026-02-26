import os
import json
import numpy as np
import onnxruntime as ort

def init():
    global session
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_dir, "model.onnx")

    session = ort.InferenceSession(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_array = np.array(data["data"]).astype(np.float32)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_array})

        predictions = outputs[0]

        return json.dumps({
            "predictions": predictions.tolist()
        })

    except Exception as e:
        return json.dumps({
            "error": str(e)
        })