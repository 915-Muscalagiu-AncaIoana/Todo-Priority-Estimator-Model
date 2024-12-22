import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.metrics import classification_report
model_path = "../todo-priority-model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load the dataset
csv_path = "test_todo_dataset.csv"
data = pd.read_csv(csv_path)

# Tokenization and Data Preparation
def prepare_data(texts, tokenizer, max_length=128):
    inputs = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return inputs

# Prepare inputs and ground truth labels
texts = data['todo_text']
labels = data['priority'].tolist()

inputs = prepare_data(texts, tokenizer)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Perform predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

# Evaluate the model
print("\nClassification Report:")
print(classification_report(labels, predictions, target_names=['low', 'high']))

# Save predictions back to CSV
data['predicted_priority'] = [p for p in predictions]
output_csv_path = "test_todo_predictions.csv"
data.to_csv(output_csv_path, index=False)
print(f"\nPredictions saved to {output_csv_path}")
