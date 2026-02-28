import torch
import torch.nn.functional as F

tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(tokenizer_, model_):
    global tokenizer, model
    tokenizer = tokenizer_
    model = model_

model.to(device)

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
    id2label = model.config.id2label

    return {
        "sentiment": id2label[pred_id.item()],
        "confidence": round(confidence.item(), 3),
    }