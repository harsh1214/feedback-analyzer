from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = BASE_DIR / "static" / "index.html"

app = FastAPI()

class Request(BaseModel):
    sentence: str
    aspect: str

@app.get('/')
def serve_index():
    return FileResponse(INDEX_FILE)

@app.post('/api/predict/')
def predict_feedback(req: Request):
    from app.predict import predict
    return predict(req.sentence, req.aspect)
