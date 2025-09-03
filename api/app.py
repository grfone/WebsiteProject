import os
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mega import Mega
import io


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://api.grfone.es"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mega.nz setup
MEGA_EMAIL = os.getenv("MEGA_EMAIL")
MEGA_PASSWORD = os.getenv("MEGA_PASSWORD")

def get_mega_client():
    mega = Mega()
    return mega.login(MEGA_EMAIL, MEGA_PASSWORD)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    try:
        # Get Mega client
        mega_client = get_mega_client()

        # Prepare file metadata
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        filename = f"{timestamp}_{file.filename}"

        # Stream file content to Mega
        file_content = await file.read()
        file_stream = io.BytesIO(file_content)

        # Upload to Mega
        mega_client.uploadfile(file_stream, filename, mega_client.get_user()['id'])

        return {"filename": filename, "message": "Image uploaded to Mega.nz successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")