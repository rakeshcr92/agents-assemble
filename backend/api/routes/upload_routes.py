from fastapi import APIRouter, UploadFile, File, HTTPException, status,FastAPI
from typing import List
import uuid
from pathlib import Path


router = APIRouter()

# --- Configuration ---
UPLOAD_DIRECTORY = Path("uploaded_files")
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- Helper Functions ---

def allowed_file(filename: str) -> bool:
    """Checks if the file extension is allowed."""
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
# --- API Endpoints ---

@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Uploads a single file to the server.
    Performs validation for file type and size.
    Saves the file with a unique name.
    """
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read the file content in chunks to handle large files efficiently
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB."
        )

    # Generate a unique filename to prevent overwrites
    original_filename = Path(file.filename).name
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIRECTORY / unique_filename

    try:
        with open(file_path, "wb") as f:
            f.write(contents)
    except IOError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file: {e}"
        )

    return {
        "message": "File uploaded successfully",
        "filename": unique_filename,
        "original_filename": original_filename,
        "file_size": len(contents),
        "file_path": str(file_path)
    }

@router.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    """
    Uploads multiple files to the server.
    Iterates through each file, performing validation and saving.
    """
    uploaded_files_info = []
    for file in files:
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed for {file.filename}. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File {file.filename} too large. Maximum size is {MAX_FILE_SIZE_MB} MB."
            )

        original_filename = Path(file.filename).name
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIRECTORY / unique_filename

        try:
            with open(file_path, "wb") as f:
                f.write(contents)
            uploaded_files_info.append({
                "filename": unique_filename,
                "original_filename": original_filename,
                "file_size": len(contents),
                "file_path": str(file_path),
                "status": "success"
            })
        except IOError as e:
            uploaded_files_info.append({
                "filename": original_filename,
                "status": "failed",
                "error": f"Could not save file: {e}"
            })

    return {
        "message": f"Successfully processed {len(uploaded_files_info)} files.",
        "uploaded_files": uploaded_files_info
    }
