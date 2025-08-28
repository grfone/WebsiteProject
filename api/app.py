from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for www.grfone.es
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.grfone.es"],  # Allow your webpage
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model (e.g., scikit-learn model)
model = joblib.load("api/model.pkl")

# Define input schema
class InputData(BaseModel):
    input: list[float]  # Adjust based on your model's input shape

# Prediction endpoint
@app.post("/predict")
async def predict(data: InputData):
    input_array = np.array([data.input])  # Convert to numpy array
    prediction = model.predict(input_array)
    return {"result": prediction.tolist()}