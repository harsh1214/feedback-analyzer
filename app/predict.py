import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / ".." / "model"

tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))
model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.to(device)

id2label = model.config.id2label

def predict(sentence: str, aspect: str):
    text = sentence + " [SEP] " + aspect

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)

    confidence, pred_id = torch.max(probs, dim=1)

    return {
        "sentiment": id2label[pred_id.item()],
        "confidence": round(confidence.item(), 3),
        "probabilities": {
            id2label[i]: round(probs[0][i].item(), 3) for i in range(len(id2label))
        }
    }