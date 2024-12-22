from fastapi import FastAPI
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

model_path = "./todo-priority-model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

@app.post("/predict")
async def predict(todo_text: str):

    inputs = tokenizer(todo_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        priority = torch.argmax(probabilities, dim=1).item() + 1

    return {"priority": priority}
