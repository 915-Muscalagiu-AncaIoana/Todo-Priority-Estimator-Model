# Install necessary packages
# pip install fastapi uvicorn transformers torch

from fastapi import FastAPI
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# Load model and tokenizer
model_path = "./todo-priority-model"  # Path to your fine-tuned model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

@app.post("/predict")
async def predict(todo_text: str):
    # Tokenize input
    inputs = tokenizer(todo_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        priority = torch.argmax(probabilities, dim=1).item() + 1  # Priorities 1, 2, 3, 4

    return {"priority": priority}
