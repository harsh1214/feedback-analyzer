from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from app.predict import predict
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_PATH = BASE_DIR / "static"

app = FastAPI()

app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")
class Request(BaseModel):
    sentence: str
    aspect: str

tokenizer = None
model = None
MODEL_NAME = "harsh1214/feedback-analyzer"

@app.on_event("startup")
async def startup_event():
    global tokenizer, model
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

@app.get('/')
def root():
    return FileResponse(STATIC_PATH / "index.html")

@app.post('/api/predict/')
def predict_feedback(req: Request):
    return predict(req.sentence, req.aspect)
