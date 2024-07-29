# Last Updated Date: 2024-07-29
# Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
# Purpose: Fine-tuning the model based on intents.json
# Status: INITIAL DEVELOPMENT (ONGOING), haven't ran or debugged it yet.
# Personal Note: Use cs224n_dfp conda env

import json #step 1
from typing import List, Any
import numpy as np #step 3-2
from sklearn.model_selection import KFold #step 3-2
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch

"""
Step 1: 
Load the intents.json file

"""

intents_path = '/Users/turtle/Projects_Python/KVAI/code/beta/intents.json'

with open(intents_path, 'r') as file:
    intents_data = json.load(file) #intents_data would be a LIST of dictionaries here.

"""
Step 2: 
Load the tokenizer & model. 
For this chatbot, we will be using DistilBertTokenizer & DistilBertForSequenceClassifciation.

- Why DistilBERRT?
Way less parameters with similar level of performance.

- Why SequenceClassification?
We are using DistilBERT to identify the intent of the user.
Hence, SequenceClassification is more suitable than the regular DistilBERT model.

- Why base?
No need for our model to be big in size.

- Why uncased?
No need for our model to differentiate between uppercase and lowercase.

"""

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

"""
Step 3:
Prepare the dataset.

->Step 3-1:
Parse every intent and the possible user inputs that result from such an intent.


(Resolved) Make sure to encode the labels so that our labels would be a number instead of str
[ATTENTION NEEDED] make sure that we are accessing the intent dictionary correctly!!! 
"""


data = []
labels = []
for intent in intents_data: #accessing each intent dictionary
    for text in intent['text']:
        data.append(text)
        labels.append(intent['intent'])

"""
->Step 3-2:
Create ids for labels so that we can use them for predictions.

(Resolved) Why not label_dict.values() but label_ids = [label_dict[label] for label in labels]?
-> it's because we need to create an id that correspond to each label in the labels list
instead of just getting a set of labels!!! these label_ids should correspond to each text in data so that
we can use it for calculating the accuracy in evaluate stage. 

"""

label_dict = {}
for idx, label in enumerate(set(labels)):
    label_dict[label] = idx

label_ids = [label_dict[label] for label in labels]


"""
->Step 3-3:
Get ready to split the data set using kfold. 
Will use the splits in step 7 when we train and evaluate the model.

- Why KFold instead of regular train_test_split?
Small dataset (i.e. small intents file). Hence, we do Kfold and make the most use of the small training data we have.
<Reference documentation>: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

"""

# Convert data and labels to np arrays 
# KFold requires np arrays
data_np = np.array(data)
labels_np = np.array(label_ids)

# define k-fold cross-validation
# why 5 splits -> default number & at least one parameter needed
kf = KFold(n_splits=5)  


"""
Step 4: 
Define a tokenizer function.
Turn off the padding cuz we will be adding a data collator with padding later on.

"""

def tokenize_function(data: Any) -> Any:
    return tokenizer(data, padding=False, truncation=True, return_tensors='pt')


"""
Step 5:
Define a data collator function to add dynamic padding. 
Reference documentation:
https://huggingface.co/learn/nlp-course/en/chapter3/2?fw=pt#dynamic-padding

Will use this in the train & evaluate function in step 7.
"""

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

"""
Step 6:
Define compute_metrics function.
"""

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = (predictions == labels).astype(np.float32).mean().item()
    f_1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1": f_1}
    
"""
Step 7:
Define train & evaluate function.
We will call this function for every (train_idx, val_idx) tuple produced by kf.split(data_np)

Reference code:
https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter3/section3.ipynb#scrollTo=4J4vOW5y7TfR

Reference documentation:
https://huggingface.co/learn/nlp-course/chapter3/3?fw=pt#training

"""
def train_evaluate(train_data, train_labels, val_data, val_labels):

    #encode the data
    train_encodings = tokenize_function(train_data)
    val_encodings = tokenize_function(val_data)

    #feed the data into torch dataset function
    train_dataset = torch.utils.data.TensorDataset(
        train_encodings['input_ids'], torch.tensor(train_labels) 
    )

    val_dataset = torch.utils.data.TensorDataset(
        val_encodings['input_ids'], torch.tensor(val_labels)
    )

    #define the dataloaders to create batches later on in training

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        val_dataset, batch_size=8, collate_fn=data_collator
    )

    #define training arguments
    training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

    #define the trainer
    trainer = Trainer(
        model,
        training_args,
        train_dataset= train_dataset,
        eval_dataset= val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    #run train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    model.save_pretrained('./distilbert-finetuned')
    tokenizer.save_pretrained('./distilbert-finetuned')


"""
Step 7:
Run a for loop on every (train_idx, val_idx) tuple produced by kf.split(data_np).

"""

for train_idx, val_idx in kf.split(data_np):
    train_data, val_data = data_np[train_idx], data_np[val_idx]
    train_labels, val_labels = labels_np[train_idx], labels_np[val_idx]
    train_evaluate(train_data, train_labels, val_data, val_labels)


"""
Notes:
[ATTENTION NEEDED] 1) 
for intent file, we only use intent & text -- 
we don't need other stuff

[ATTENTION NEEDED] 2) Sklearn not being implemented for some reason -- "pip install sklearn"

[RESOLVED] 3)  labels -> currently str but need to convert it to numbers
label_dict = {label: idx for idx, label in enumerate(set(labels))}
label_ids = [label_dict[label] for label in labels]

[ATTENTION NEEDED] 4) Need to transfer this file to Colab
-> Consider uploading it to Kaggle? TBD

"""