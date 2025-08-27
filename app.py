import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "./hf_cache"

app = FastAPI()
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="./hf_cache")
model = model.half() if torch.cuda.is_available() else model

class TextPair(BaseModel):
    text1: str
    text2: str

def similarity_score(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    score = (score + 1) / 2  
    return round(score, 3)

@app.post("/")
async def get_similarity(data: TextPair):
    score = similarity_score(data.text1, data.text2)
    return {"similarity score": score}
