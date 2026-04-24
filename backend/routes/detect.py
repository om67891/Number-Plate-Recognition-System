import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from backend.model.yolo_ocr import detect_and_read
from backend.utils.storage import get_history, get_stats, save_plate, vehicles_today

router = APIRouter()


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    file_path = Path(f"temp_{file.filename}")
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        plate_text = detect_and_read(str(file_path))
        result = {
            "plate_text": plate_text,
            "timestamp": str(datetime.now()),
            "duplicate": False,
        }

        save_plate(result)
        return result
    finally:
        await file.close()
        if file_path.exists():
            file_path.unlink()


@router.get("/history")
def history():
    return get_history()


@router.get("/stats")
def stats():
    return get_stats()


@router.get("/today")
def today():
    return vehicles_today()
