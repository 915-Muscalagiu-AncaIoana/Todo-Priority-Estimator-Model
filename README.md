# TODO Prioritization Model

This repository contains the code and resources for training, fine-tuning, and deploying a model to prioritize TODO comments in code. The solution includes data collection, model training, and an API endpoint for inference.

## Repository Structure

- `TODO_Finetuning.ipynb`: This Jupyter notebook is used for training the TODO prioritization model. It utilizes the dataset loader to load and preprocess the data and the `Trainer` class for fine-tuning the model.
- `crawler.py`: This script is used to collect TODO comments from GitHub repositories. Run this script to start the crawler, inputting your GitHub API key and Comet ML API key for data access and logging.
- `backend.py`: This file contains the backend code for deploying the inference endpoint. You can start the endpoint using Uvicorn to serve predictions on new TODO comments.

## Getting Started

### Prerequisites
1. **Python**: Make sure you have Python installed (preferably Python 3.9 or higher).
2. **Dependencies**: Install the required packages by running:
   ```bash
   pip install -r requirements.txt
