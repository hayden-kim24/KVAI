import json
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from utils import load_json


# Load data
case_data = load_json('/Users/turtle/Projects_Python/KVAI/code/beta/sample_cases.json')
intents = load_json('/Users/turtle/Projects_Python/KVAI/code/beta/intents.json')

# Load fine-tuned model

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Map intents to responses
intent_map = {intent['intent']: intent['responses'] for intent in intents}

def predict_intent(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

def find_case(policestation, firnumber, year):
    for case in case_data:
        if (case['Police Station'] == policestation and
            case['FIR Number'] == firnumber and
            case['Year'] == year):
            return case
    return None

def answer_question(case, intent):
    if intent == 'Case Type':
        return f"The case type is {case['Case Type']}."
    elif intent == 'Case Status':
        return f"The case status is {case['Case Stage']}."
    elif intent == 'Next Hearing Date':
        return f"The next hearing date is {case['Next Hearing Date']}."
    else:
        return "Sorry, I don't have information on that."

def chatbot_response(user_input, policestation, firnumber, year):
    case = find_case(policestation, firnumber, year)
    if not case:
        return "Case not found. Please check the details or contact International Bridges to Justice at 1-800-555-5555."

    intent_id = predict_intent(user_input)
    intent = list(intent_map.keys())[intent_id]
    
    response = answer_question(case, intent)
    return response

if __name__ == '__main__':
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        policestation = input("What is the name of the Police Station?: ")
        firnumber = int(input("What is the FIR Number?: "))
        year = int(input("What is the Year for the case?: "))
        
        response = chatbot_response(user_input, policestation, firnumber, year)
        print(f"Bot: {response}")
        
        continue_query = input("Do you have any other questions? (yes/no): ")
        if continue_query.lower() != 'yes':
            break
