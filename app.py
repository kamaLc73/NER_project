from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pretrained model and vocabularies
model = tf.keras.models.load_model("saves/model_pretrained.keras")

with open("saves/word_vocab.json", "r", encoding="utf-8") as f:
    word_vocab = json.load(f)

with open("saves/tag_vocab.json", "r", encoding="utf-8") as f:
    tag_vocab = json.load(f)

reverse_tag_vocab = {v: k for k, v in tag_vocab.items()}
max_len = model.input_shape[1]  # Get max_len from the model input shape

# Initialize FastAPI app
app = FastAPI()

# Mount static files (for CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

class SentenceInput(BaseModel):
    sentence: str

def predict_entities(sentence: str):
    words = sentence.split()
    encoded_sentence = [word_vocab.get(word, word_vocab["<UNK>"]) for word in words]
    padded_sentence = pad_sequences([encoded_sentence], maxlen=max_len, padding="post")
    predictions = model.predict(padded_sentence)

    predicted_tags = []
    for prediction in predictions[0]:
        tag_index = np.argmax(prediction)
        predicted_tags.append(reverse_tag_vocab.get(tag_index))

    aligned_tags = predicted_tags[:len(words)]

    return list(zip(words, aligned_tags))

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
def predict(sentence: str = Form(...)):
    try:
        entities = predict_entities(sentence)
        return {"sentence": sentence, "entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
