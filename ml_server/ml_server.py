import pandas as pd
import requests
from minio import Minio
from minio.error import S3Error
import torch
import io
import json
from os import environ
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

bert_model_name = 'bert-base-uncased'
num_classes = 5
max_length = 128
batch_size = 16
num_epochs = 1
learning_rate = 2e-5

db_server_url = environ.get('DB_SERVER_URL')
bucket_name = environ.get('MINIO_BUCKET')
endpoint = environ.get('MINIO_ENDPOINT')
access_key = environ.get('MINIO_ACCESS_KEY')
secret_key = environ.get('MINIO_SECRET_KEY')
model_file_name = "model_weights.pt"
encoding_file_name = "encodings.json"

client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)


@app.route('/train', methods=['POST'])
def train():
    app.logger.info("Retrieving training data ...")
    response = requests.get(db_server_url + '/news')
    data = pd.DataFrame(response.json()['news'], columns=['text', 'label'])
    app.logger.info("Retrieved training data.")

    data['labelId'] = data['label'].factorize()[0]
    train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'].to_list(), data['labelId'].to_list(), test_size=0.2,
                                                                        random_state=42)

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = BERTClassifier(bert_model_name, num_classes)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    app.logger.info("Start model training ...")
    for epoch in range(num_epochs):
        app.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        train_model(model, train_dataloader, optimizer, scheduler)
        app.logger.info("Evaluating ...")
        accuracy, report = evaluate(model, val_dataloader)
        app.logger.info(f"Validation Accuracy: {accuracy:.4f}")
        print(report)
    app.logger.info("Model trained successfully ...")

    app.logger.info("Model saving ...")
    save_model(model, model_file_name)
    save_encoding(data)
    app.logger.info("Model saved successfully ...")

    return Response(status=204)


@app.route('/predict', methods=['GET'])
def predict():
    request_data = request.get_json()
    text = request_data["text_to_predict"]

    app.logger.info("Model retrieving ...")
    model = get_model(model_file_name)
    app.logger.info("Model retrieved successfully ...")

    model.eval()
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    app.logger.info("Predicting ...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predictions = torch.max(outputs, dim=1)
    app.logger.info("Predicted successfully.")

    encoding = get_encoding()
    if encoding[str(predictions.item())]:
        response = {'prediction': encoding[str(predictions.item())]}
        return jsonify(response), 200
    else:
        response = {'prediction': 'unknown'}
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
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        #self.dropout = nn.Dropout(0.1)
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
    for i, batch in enumerate(data_loader):
        app.logger.info(f'Processing batch {i+1}/{len(data_loader)}')
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()


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


def save_model(model, model_name):
    torch.save(model, model_name)
    try:
        app.logger.info(f"Saving model to {bucket_name}/{model_name} ...")
        if not client.bucket_exists(bucket_name):
            app.logger.info(f"Bucket {bucket_name} doesn't exist")
            client.make_bucket(bucket_name)

        client.fput_object(bucket_name, model_name, model_name)
        app.logger.info(f"Model successfully uploaded to {bucket_name}/{model_name}")
    except S3Error as e:
        app.logger.error(f"Error uploading model: {e}")


def save_encoding(dataset):
    data_dict = dataset.set_index('labelId')['label'].to_dict()
    json_str = json.dumps(data_dict)
    json_bytes = io.BytesIO(json_str.encode('utf-8'))
    try:
        app.logger.info(f"Saving encoding to {bucket_name}/{encoding_file_name} ...")
        if not client.bucket_exists(bucket_name):
            app.logger.info(f"Bucket {bucket_name} doesn't exist")
            client.make_bucket(bucket_name)

        client.put_object(bucket_name, encoding_file_name, json_bytes, length=len(json_str), content_type='application/json')
        app.logger.info(f"Encoding successfully uploaded to {bucket_name}/{encoding_file_name}")
    except S3Error as e:
        app.logger.error(f"Error uploading encoding: {e}")


def get_model(model_name):
    try:
        if not client.bucket_exists(bucket_name):
            app.logger.warn(f"Bucket '{bucket_name}' does not exist.")
        else:
            model_data = client.get_object(bucket_name, model_name).read()
            model_buffer = io.BytesIO(model_data)
            model = torch.load(model_buffer)
            return model
    except S3Error as e:
        app.logger.error(f"Error: {e}")


def get_encoding():
    try:
        if not client.bucket_exists(bucket_name):
            app.logger.warn(f"Bucket '{bucket_name}' does not exist.")
        else:
            encoding = client.get_object(bucket_name, encoding_file_name).read()
            json_buffer = io.BytesIO(encoding)
            data_dict = json.load(json_buffer)
            return data_dict
    except S3Error as e:
        app.logger.error(f"Error: {e}")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=environ.get('ML_SERVER_PORT'))