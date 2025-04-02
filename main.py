from fastapi import FastAPI, File, UploadFile
from vlm_classes.molmo import molmo_handler
import shutil
import os

app = FastAPI()
model_handler = molmo_handler()

@app.post("/generate-text/")
async def generate_text(file: UploadFile = File(...), prompt: str = ""):

    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Generate text
    generated_text = model_handler.generate_text(temp_file_path, prompt)
    
    os.remove(temp_file_path)
    
    return {"generated_text": generated_text}
