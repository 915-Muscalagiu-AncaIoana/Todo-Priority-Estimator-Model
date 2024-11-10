import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, DataCollatorWithPadding, \
    Trainer
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import numpy as np
import random


# Set a fixed seed for reproducibility
def set_seed(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Set the seed
set_seed(50)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    probs = np.exp(pred.predictions) / np.sum(np.exp(pred.predictions), axis=1, keepdims=True)
    loss = log_loss(labels, probs)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cross_entropy_loss': loss
    }


class TodoClassifierTrainer:
    def __init__(self, experiment, dataset, tokenizer_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(tokenizer_name, num_labels=3)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.data = dataset
        self.experiment = experiment
        self.train_dataset, self.val_dataset = self._prepare_datasets()

    def _prepare_datasets(self):
        # Split dataset into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.data["todo_text"].tolist(),
            self.data["priority"].tolist(),
            test_size=0.2,
            random_state=42
        )

        # Convert to Hugging Face Dataset
        train_data = Dataset.from_pandas(pd.DataFrame({"text": train_texts, "label": train_labels}))
        val_data = Dataset.from_pandas(pd.DataFrame({"text": val_texts, "label": val_labels}))

        # Tokenize dataset
        train_data = train_data.map(self._tokenize_function, batched=True)
        val_data = val_data.map(self._tokenize_function, batched=True)

        return train_data, val_data

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def train_model(self):
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=4,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=2,
            report_to="comet_ml",
            seed=50
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics
        )

        trainer.train()

        # Save the model
        model_path = "./todo-priority-model"
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)

        # Log model as an artifact to Comet ML
        self.experiment.log_asset_folder(model_path, log_file_name="todo-priority-model")
        print("Model saved to Comet ML as an artifact.")
