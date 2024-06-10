import pandas as pd
import requests
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import torch
import json
from os import environ
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

db_server_url = environ.get('DB_SERVER_URL')
mlflow_tracking_url = environ.get('MLFLOW_TRACKING_URI')
params_file_name = "params.json"
encoding_file_name = "encodings.json"
model_artifact_name = "bert_model"
experiment_name = "bert-baseline-exp-1"

mlflow.set_tracking_uri(mlflow_tracking_url)
mlflow.set_experiment(experiment_name)


@app.route('/train', methods=['POST'])
def train():
    app.logger.info("Retrieving training data ...")
    response = requests.get(db_server_url + '/news')
    data = pd.DataFrame(response.json()['news'], columns=['text', 'label'])
    app.logger.info("Retrieved training data.")

    data['labelId'] = data['label'].factorize()[0]
    train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'].to_list(),
                                                                        data['labelId'].to_list(), test_size=0.2,
                                                                        random_state=42)

    params = {
        'bert_model_name': 'bert-base-uncased',
        'num_classes': 5,
        'max_length': 64,
        'batch_size': 16,
        'num_epochs': 1,
        'learning_rate': 2e-5,
    }

    tokenizer = BertTokenizer.from_pretrained(params['bert_model_name'])
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, params['max_length'])
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, params['max_length'])
    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'])

    model = BERTClassifier(params['bert_model_name'], params['num_classes'])

    optimizer = AdamW(model.parameters(), lr=params['learning_rate'])
    total_steps = len(train_dataloader) * params['num_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    app.logger.info("Start model training ...")
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.set_tag("Training Info", "Basic BERT model for news data")
        input_ids, attention_mask, outputs = None, None, None
        for epoch in range(params['num_epochs']):
            app.logger.info(f"Epoch {epoch + 1}/{params['num_epochs']}")
            input_ids, attention_mask, outputs = train_model(model, train_dataloader, optimizer, scheduler)
            app.logger.info("Evaluating ...")
            accuracy, report = evaluate(model, val_dataloader)
            app.logger.info(f"Validation Accuracy: {accuracy:.4f}")
            mlflow.log_metric("accuracy", accuracy)
            print(report)
        app.logger.info("Model trained successfully ...")

        input_example = {
            "input_ids": input_ids[0].numpy().tolist(),
            "attention_mask": attention_mask[0].numpy().tolist()
        }
        signature = infer_signature(input_example, outputs[0].detach().numpy())

        encodings = data.set_index('labelId')['label'].to_dict()
        with open(encoding_file_name, 'w') as f:
            json.dump(encodings, f)

        mlflow.log_artifact(encoding_file_name)

        mlflow.pytorch.log_model(model,
                                 artifact_path=model_artifact_name,
                                 signature=signature,
                                 input_example=input_example)

        return Response(status=204)


@app.route('/predict', methods=['GET'])
def predict():
    request_data = request.get_json()
    text = request_data["text_to_predict"]

    app.logger.info("Last run retrieving ...")
    client = MlflowClient()
    experiments = client.search_experiments(filter_string=f"name = '{experiment_name}'", max_results=1)
    runs = client.search_runs(experiment_ids=[experiments[0].experiment_id])
    if runs and len(runs) > 0:
        last_run_id = runs[0].info.run_id

        app.logger.info("Model retrieving ...")
        model_uri = f"runs:/{last_run_id}/{model_artifact_name}"
        model = mlflow.pytorch.load_model(model_uri)
        app.logger.info("Model retrieved successfully ...")

        app.logger.info("Encoding retrieving ...")
        response = requests.get(mlflow.get_artifact_uri(encoding_file_name))
        encoding = response.json()
        app.logger.info("Encoding retrieved successfully ...")

        model.eval()
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        app.logger.info("Predicting ...")
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predictions = torch.max(outputs, dim=1)
        app.logger.info("Predicted successfully.")

        if encoding[str(predictions.item())]:
            response = {'prediction': encoding[str(predictions.item())]}
            return jsonify(response), 200
        else:
            response = {'prediction': 'unknown'}
            return jsonify(response), 500
    else:
        response = {'message': 'Model not found'}
        return jsonify(response), 500


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length',
                                  truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)}


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        # self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        # x = self.dropout(pooled_output)
        # logits = self.fc(x)
        return logits


def train_model(model, data_loader, optimizer, scheduler):
    model.train()
    input_ids, attention_mask, outputs = None, None, None
    for i, batch in enumerate(data_loader):
        app.logger.info(f'Processing batch {i + 1}/{len(data_loader)}')
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

    return input_ids, attention_mask, outputs


def evaluate(model, data_loader):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=environ.get('ML_SERVER_PORT'))
