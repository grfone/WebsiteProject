import os
import zipfile
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.grfone.es"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Unzip and load models
models_dir = "models"
models = []

for i in range(1, 6):  # model_fold1 ... model_fold5
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
    input_array = np.array(data.input, dtype=np.float32)

    # Ensure input has correct shape [1,380,380,3]
    if input_array.ndim == 2:  # flattened
        size = int(np.sqrt(len(input_array[0]) / 3))
        input_array = np.reshape(input_array, (1, size, size, 3))

    summed_output = None
    for model in models:
        output = model.predict(input_array, verbose=0)
        summed_output = output if summed_output is None else summed_output + output

    prediction = summed_output.tolist()
    class_idx = int(np.argmax(summed_output))
    return {"raw_output": prediction, "class_idx": class_idx}
