import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "harsh1214/feedback-analyzer"

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()

def predict(sentence: str, aspect: str):
    load_model()
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
    id2label = model.config.id2label

    return {
        "sentiment": id2label[pred_id.item()],
        "confidence": round(confidence.item(), 3),
    }