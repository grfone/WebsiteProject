import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from threading import Lock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # api/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # repo root
MODELS_ROOT = os.path.join(ROOT_DIR, "models")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://api.grfone.es"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Runtime env tweaks (CPU-only & quieter logs)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_models = []
_models_lock = Lock()

def _resolve_model_path(i: int) -> str:
    fold_dir = os.path.join(MODELS_ROOT, f"model_fold{i}.zip")
    # Extract and check for .keras file inside the zip
    import zipfile
    with zipfile.ZipFile(fold_dir, 'r') as zip_ref:
        keras_files = [f for f in zip_ref.namelist() if f.endswith(".keras")]
        if keras_files:
            temp_dir = os.path.join(MODELS_ROOT, f"temp_fold{i}")
            os.makedirs(temp_dir, exist_ok=True)
            zip_ref.extract(keras_files[0], temp_dir)
            return os.path.join(temp_dir, keras_files[0])
    raise FileNotFoundError(f"No .keras file found in {fold_dir}")

@app.on_event("startup")
async def load_models():
    global _models
    with _models_lock:
        if _models:  # already loaded (e.g., when using --preload)
            return
        for i in range(1, 6):
            path = _resolve_model_path(i)
            _models.append(tf.keras.models.load_model(path))
    print(f"Loaded {_models.__len__()} models.")

class InputData(BaseModel):
    # flattened [1, 380*380*3] OR shaped [1,380,380,3]
    input: list

@app.get("/health")
async def health():
    return {"status": "ok", "models": len(_models)}

@app.post("/predict")
async def predict(data: InputData):
    if not _models:
        raise HTTPException(status_code=500, detail="Models not loaded")

    arr = np.array(data.input, dtype=np.float32)
    expected_size = 380 * 380 * 3

    # Accept [1, H, W, C], [H*W*C], or [[H*W*C]]
    if arr.ndim == 1 and arr.size == expected_size:
        arr = arr.reshape(1, 380, 380, 3)
    elif arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == expected_size:
        arr = arr.reshape(1, 380, 380, 3)
    elif arr.ndim == 4 and arr.shape == (1, 380, 380, 3):
        pass
    else:
        raise HTTPException(status_code=400, detail=f"Expected 1x380x380x3 (or flattened), got shape {arr.shape}")

    summed = None
    for m in _models:
        out = m.predict(arr, verbose=0)
        summed = out if summed is None else summed + out

    class_idx = int(np.argmax(summed, axis=-1)[0])
    return {"raw_output": summed.tolist(), "class_idx": class_idx}