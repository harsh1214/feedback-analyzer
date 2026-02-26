from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
from pathlib import Path
# from pydantic import BaseModel
# from app.predict import predict

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_PATH = BASE_DIR / "static" / "index.html"

app = FastAPI()

app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

# class Request(BaseModel):
#     sentence: str
#     aspect: str

@app.get('/')
def serve_index():
    return print(str(STATIC_PATH))

# @app.post('/api/predict/')
# def predict_feedback(req: Request):
#     return predict(req.sentence, req.aspect)
