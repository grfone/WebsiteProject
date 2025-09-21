import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from PIL import Image
import io

app = FastAPI()

# Add CORS middleware FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correct paths for your structure:
demo_root = Path(__file__).parent.parent  # Points to demo/
docs_root = demo_root / "docs"  # Points to demo/docs/
main_project_root = demo_root.parent  # Points to WebsiteProject/

# Get class names from main project
resources_path = main_project_root / "resources" / "TypesOfClouds"
if resources_path.exists():
    classes = sorted([d.name for d in resources_path.iterdir() if d.is_dir()])
else:
    classes = ['Altocumulus', 'Altostratus', 'Cirrocumulus', 'Cirrostratus', 'Cirrus',
               'Contrail', 'Cumulonimbus', 'Cumulus', 'Nimbostratus', 'Stratocumulus', 'Stratus']

# PRELOAD MODELS ON STARTUP
print("🔄 Loading models...")
models = []
results_path = main_project_root / "results" / "TypesOfClouds"

for i in range(1, 6):
    model_path = results_path / f"model_fold{i}.keras"
    if model_path.exists():
        models.append(tf.keras.models.load_model(str(model_path)))
        print(f"✅ Loaded model: model_fold{i}.keras")
    else:
        print(f"⚠️  Warning: Model not found: {model_path}")

if not models:
    print("❌ ERROR: No models found! Server will not work properly.")
else:
    print(f"🎉 All {len(models)} models loaded successfully!")


def center_crop_image(image_path, target_size=(380, 380)):
    """Center crop the image to a square, then resize to target_size"""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Get the minimum dimension for square cropping
        width, height = img.size
        size = min(width, height)

        # Calculate center coordinates for cropping
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size

        # Crop to square
        cropped_img = img.crop((left, top, right, bottom))

        # Resize to target size (PIL returns uint8 values 0-255 by default)
        final_img = cropped_img.resize(target_size, Image.Resampling.LANCZOS)

        return final_img


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    try:
        ext = file.filename.split(".")[-1]
        filename = f"{uuid.uuid4()}.{ext}"

        # Save uploads in demo/ directory
        uploads_dir = demo_root / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        file_path = uploads_dir / filename

        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Use preloaded models
        if not models:
            raise HTTPException(status_code=500, detail="No models loaded")

        # Get input size from first model
        input_size = models[0].input_shape[1:3]

        # Center crop and resize image (matching JavaScript logic)
        cropped_img = center_crop_image(file_path, target_size=(380, 380))

        # Convert to numpy array - KEEP VALUES AS 0-255 to match JavaScript
        img_array = np.array(cropped_img, dtype=np.float32)  # Values: 0-255 as float32
        img_array = np.expand_dims(img_array, axis=0)  # Shape: [1, 380, 380, 3]

        # Ensemble predictions
        preds_sum = np.zeros((1, len(classes)))
        for model in models:
            preds = model.predict(img_array, verbose=0)
            preds_sum += preds

        pred_class_idx = np.argmax(preds_sum, axis=1)[0]
        pred_class = classes[pred_class_idx]
        confidence = float(preds_sum[0][pred_class_idx] / len(models))

        # Clean up
        file_path.unlink()

        print(f"🌥️  Prediction: {pred_class} (confidence: {confidence:.2f})")
        return {"filename": filename, "prediction": pred_class, "confidence": confidence}

    except Exception as e:
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Static files AFTER API routes
app.mount("/", StaticFiles(directory=str(docs_root), html=True), name="static")
app.mount("/images", StaticFiles(directory=str(docs_root / "images")), name="images")


@app.get("/")
async def root():
    index_path = docs_root / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        available_files = [f.name for f in docs_root.iterdir() if f.is_file() and f.suffix == '.html']
        raise HTTPException(status_code=404, detail=f"index.html not found. Available: {available_files}")


@app.get("/types-of-clouds")
@app.get("/types_of_clouds.html")
async def types_of_clouds():
    html_path = docs_root / "types_of_clouds.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    else:
        available_files = [f.name for f in docs_root.iterdir() if f.is_file() and f.suffix == '.html']
        raise HTTPException(status_code=404, detail=f"types_of_clouds.html not found. Available: {available_files}")


@app.get("/images/{filename:path}")
async def get_image(filename: str):
    image_path = docs_root / "images" / filename
    if image_path.exists():
        return FileResponse(str(image_path))
    else:
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)