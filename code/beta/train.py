import json
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Load data
with open('intent.json') as f:
    intents = json.load(f)

# Prepare dataset
data = []
labels = []
for intent in intents:
    for text in intent['text']:
        data.append(text)
        labels.append(intent['intent'])

# Encode labels
label_dict = {label: idx for idx, label in enumerate(set(labels))}
label_ids = [label_dict[label] for label in labels]

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(data, label_ids, test_size=0.1)

# Load tokenizer and encode texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

train_encodings = encode_texts(train_texts)
val_encodings = encode_texts(val_texts)

train_dataset = torch.utils.data.TensorDataset(train_encodings.input_ids, torch.tensor(train_labels))
val_dataset = torch.utils.data.TensorDataset(val_encodings.input_ids, torch.tensor(val_labels))

# Define model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_dict))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train model
trainer.train()

# Save model
model.save_pretrained('./bert-finetuned')
tokenizer.save_pretrained('./bert-finetuned')
