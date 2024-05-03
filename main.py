from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from kobert_model import BERTClassifier, bertmodel  # KoBERT 모델 클래스를 임포트

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

# 모델 인스턴스화 및 가중치 로드
model = BERTClassifier(bertmodel,  dr_rate=0.5)
model.load_state_dict(torch.load("kobert.pt", map_location=torch.device('cpu')))
model.eval()