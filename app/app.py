from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from app.predict import predict, load_model

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = BASE_DIR / "static" / "index.html"

MODEL_NAME = "harsh1214/feedback-analyzer"


app = FastAPI()

class Request(BaseModel):
    sentence: str
    aspect: str

@app.on_event("startup")
def startup_event():
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_NAME))
    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_NAME))
    model.eval()
    load_model(tokenizer, model)


@app.get('/')
def serve_index():
    return FileResponse(INDEX_FILE)

@app.post('/api/predict/')
def predict_feedback(req: Request):
    return predict(req.sentence, req.aspect)
