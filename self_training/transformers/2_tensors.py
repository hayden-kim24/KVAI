# Last Updated Date: 2024-07-17
# Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
# Purpose: Self-Training (Not to be used for actual app)
# Status: Done
# Based on: Hugging Face NLP Course Chapter 2, https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt
# Code is directly from Hugging Face NLP Course website

from transformers import AutoTokenizer, pipeline, AutoModel, AutoModelForSequenceClassification
import torch

classifier = pipeline("sentiment-analysis")
a = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
print(a)

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModel.from_pretrained(checkpoint)
# outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)
# print(outputs)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits)
checkpoint = "distilbert-base-uncased"  # Use base model name for tokenizer

raw_data = [
    "I don't like the current judicial system.",
    "Wow it is amazing that there is a chatbot that assists prisoners",
]

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenize the raw text data
inputs = tokenizer(
    raw_data,
    padding=True,
    truncation=True,
    return_tensors="pt"  # Return PyTorch tensors
)

print(inputs)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
final_predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(final_predictions)
print(model.config.id2label)

