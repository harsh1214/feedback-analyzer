# 🍽️ NLP Feedback Analyzer for Restaurants (Aspect-Based Sentiment Analysis)

An end-to-end **Aspect-Based Sentiment Analysis (ABSA)** system for restaurant feedback, built using **Transformers (DistilBERT)** and deployed with **FastAPI** and a lightweight **HTML + Tailwind CSS UI**.

This project allows users to enter a **review sentence** and a specific **aspect** (e.g., food, service, price) and receive a **sentiment prediction with confidence**, powered by a fine-tuned DistilBERT model.

---

## 🚀 Key Features

- Aspect-aware sentiment analysis (not generic sentiment)
- Transformer-based model (DistilBERT)
- Real-time inference via FastAPI
- Simple browser-based UI
- GPU-accelerated inference (if available)
- Clean separation of frontend, backend, and ML logic

---

## 🧠 Problem Statement

Traditional sentiment analysis answers:
> *Is this review positive or negative?*

This project answers:
> *What is the sentiment **about a specific aspect** in the review?*

### Example
**Sentence:**  
`The food was great but the service was slow`

**Aspect:**  
`service`

**Prediction:**  
`Negative (high confidence)`

---

## 🧪 Model Overview

- **Model:** `distilbert-base-uncased`
- **Task:** 4-class Aspect-Based Sentiment Classification
- **Classes:**
  - `positive`
  - `negative`
  - `neutral`
  - `conflict`
- **Training Framework:** HuggingFace Transformers
- **Inference:** PyTorch

The model was fine-tuned using the input format:

```
Sentence [SEP] Aspect
```

This enables the model to focus on sentiment relevant to the given aspect.

---

## 🧱 System Architecture

```
Browser UI (HTML + Tailwind)
        ↓
FastAPI Backend
        ↓
Tokenizer (DistilBERT)
        ↓
Fine-tuned DistilBERT Model
        ↓
Sentiment + Confidence
```

---

## 📁 Project Structure

```
.
├── app/
│   ├── app.py            # FastAPI app & routes
│   └── predict.py        # Model loading & inference logic
├── static/
│   └── index.html        # Frontend UI
├── model/ (or HuggingFace Hub)
│   └── DistilBERT model files
├── README.md
```

---

## ⚙️ API Endpoints

### Serve UI
```
GET /
```
Serves the frontend HTML page.

---

### Predict Sentiment
```
POST /api/predict/
```

#### Request Body
```json
{
  "sentence": "The food was great but the service was slow",
  "aspect": "service"
}
```

#### Response
```json
{
  "sentiment": "negative",
  "confidence": 0.82
}
```

---

## 🖥️ Frontend UI

- Built using plain HTML + Tailwind CSS
- Allows users to:
  - Enter a sentence
  - Enter an aspect
  - View sentiment and confidence instantly
- Communicates with backend via Fetch API

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **HuggingFace Transformers**
- **DistilBERT**
- **FastAPI**
- **Pydantic**
- **HTML + Tailwind CSS**
- **CUDA (optional, for GPU acceleration)**

---

## 🧠 Design Decisions

- **DistilBERT** chosen over full BERT for faster inference with minimal accuracy loss
- Model loaded **once at startup** for performance
- HuggingFace `from_pretrained()` used for portability and deployment
- UI kept minimal to emphasize ML functionality, not frontend complexity

---

## 🚧 Limitations

- `conflict` class performance is limited due to severe class imbalance
- Aspect extraction is not automated (user provides aspect)
- No authentication or rate limiting (demo-focused)

---

## 🔮 Future Improvements

- Automatic aspect extraction
- Confidence thresholding & fallback responses
- Model explainability (attention visualization)
- Dockerization & cloud deployment
- Improved handling of minority classes

---

## 👤 Author

**Harsh Yadav**  
NLP / Machine Learning Enthusiast

---

## 📜 License

This project is for educational and demonstration purposes.