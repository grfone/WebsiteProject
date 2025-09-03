import os
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import dropbox

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://api.grfone.es"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dropbox setup
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"status": "ok", "message": "API running"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    try:
        # Prepare file metadata
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        filename = f"{timestamp}_{file.filename}"
        dropbox_path = f"/{filename}"

        # Read file content
        file_content = await file.read()

        # Upload to Dropbox
        dbx.files_upload(file_content, dropbox_path, mode=dropbox.files.WriteMode("add"))

        return {"filename": filename, "message": "Image uploaded to Dropbox successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
