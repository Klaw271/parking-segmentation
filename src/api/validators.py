from fastapi import UploadFile, HTTPException

def validate_image(image_file: UploadFile):
    if image_file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

def validate_json(file: UploadFile):
    if not file.filename.endswith(".json"):
        raise HTTPException(400, "File must be .json")
    
    content = file.file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty JSON file")

    file.file.seek(0)
