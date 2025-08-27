import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

model = SentenceTransformer("LaBSE")
model = model.half() if torch.cuda.is_available() else model

class TextPair(BaseModel):
    text1: str
    text2: str

def similarity_score(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    score = (score + 1) / 2  # normalize to 0–1
    return round(score, 3)

@app.get("/")
def home():
    return {"message": "App is running! Use POST / to get similarity score."}

@app.post("/")
async def get_similarity(data: TextPair):
    score = similarity_score(data.text1, data.text2)
    return {"similarity score": score}


@app.post("/")
async def get_similarity(data: TextPair):
    score = similarity_score(data.text1, data.text2)
    return {"similarity score": score}
