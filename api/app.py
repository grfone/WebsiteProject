import os
import zipfile
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://api.grfone.es"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
models = []


@app.on_event("startup")
async def load_models():
    global models
    models_dir = "models"
    for i in range(1, 6):
        zip_path = os.path.join(models_dir, f"model_fold{i}.zip")
        extract_dir = os.path.join(models_dir, f"uncompressed_fold{i}")
        if not os.path.exists(extract_dir):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        keras_files = [f for f in os.listdir(extract_dir) if f.endswith(".keras")]
        if not keras_files:
            raise FileNotFoundError(f"No .keras file found in {extract_dir}")
        model_path = os.path.join(extract_dir, keras_files[0])
        models.append(tf.keras.models.load_model(model_path))
    print(f"Loaded {len(models)} models.")


# Input schema
class InputData(BaseModel):
    input: list[list[float]]  # shape: [1,380,380,3] flattened or similar


@app.post("/predict")
async def predict(data: InputData):
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")

    input_array = np.array(data.input, dtype=np.float32)
    expected_size = 380 * 380 * 3
    if input_array.size != expected_size:
        raise HTTPException(status_code=400,
                            detail=f"Input array must have {expected_size} elements, got {input_array.size}")

    if input_array.ndim == 2:
        input_array = np.reshape(input_array, (1, 380, 380, 3))

    summed_output = None
    for model in models:
        try:
            output = model.predict(input_array, verbose=0)
            summed_output = output if summed_output is None else summed_output + output
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    if summed_output is None:
        raise HTTPException(status_code=500, detail="No valid predictions generated")

    prediction = summed_output.tolist()
    class_idx = int(np.argmax(summed_output))
    return {"raw_output": prediction, "class_idx": class_idx}