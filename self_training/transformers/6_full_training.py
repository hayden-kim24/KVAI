# Last Updated Date: 2024-07-29
# Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
# Purpose: Self-Training (Not to be used for actual app)
# Status: Ongoing
# Based on: Hugging Face NLP Course Chapter 3,https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt
# Code is directly from Hugging Face NLP Course website

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
#paraphrase dataset

"""
Step 1: Tokenize the dataset
"""
checkpoint =  "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation = True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched = True)
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

#remove the columns that the model won't expect
#rename the column label to labels
#return Pytorch tensors instead of lists

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2","idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

"""
Step 2: Create the Dataloader and process data
"""
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], suffle = True, batch_size = 8, collate_fn = data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size = 8, collate_fn = data_collator
)


"""
Step 3: Initialize a Model
"""
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)
for batch in train_dataloader:
    # Break after the first batch
    break
outputs = model(**batch)


"""
Step 4: Initialize the Optimizer
"""
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr = 5e-5)

"""
Step 5: Initialize the Scheduler
"""
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name = "linear", optimizer = optimizer, num_warmup_steps = 0, num_training_steps = num_training_steps
)

"""
Step 6: set up the device
"""

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

"""
Step 7: Train the model
"""

from tqdm.auto import tqdm #progress bar

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

"""
Step 8: Evaluate the model
"""

import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()


"""Accelerate by using "Accelerate" library"""

from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


