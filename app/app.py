from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
from pydantic import BaseModel
from fastapi.responses import FileResponse

print("📦 Importing app.app START")

MODEL_NAME = "harsh1214/feedback-analyzer"

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = BASE_DIR / "static" / "index.html"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Lifespan START")

    try:
        print("Importing transformers...")
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        print("transformers imported")

        print("Loading tokenizer...")
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        print("tokenizer loaded")

        print("Loading model...")
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        print("model loaded")

        from app.predict import load_model
        load_model(tokenizer, model)

    except Exception as e:
        print("STARTUP ERROR:", repr(e))
        raise e

    yield
    print("🛑 Lifespan END")

app = FastAPI(lifespan=lifespan)

class Request(BaseModel):
    sentence: str
    aspect: str

@app.get("/")
def serve_index():
    return FileResponse(INDEX_FILE)