# backend/app.py
import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
bucket_name = "uploads"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/get-upload-url")
async def get_upload_url(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    try:
        ext = file.filename.split(".")[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        path = f"{bucket_name}/{filename}"

        # Generate signed URL valid for 60 seconds
        signed_url = supabase.storage.from_(bucket_name).create_signed_url(path, 60, method="PUT")

        return {"signed_url": signed_url, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))