from transformers import AutoTokenizer,  AutoModelForSequenceClassification
import torch

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "We are trying out tokenizer",
    "Tokenizer is super simple to implement",

]

# #overall tokenization
# inputs = tokenizer(raw_inputs, padding = True, truncation= True, return_tensors = "pt")
# print(inputs)

# #step by step
# #step 1: create tokens (word by word)
# tokens = tokenizer.tokenize(raw_inputs)
# print("tokens: ", tokens)

# #step 2: convert tokens into token ids
# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print("token ids: ", token_ids)

# #step 3: create attention mask
# decoded_string = tokenizer.decode(token_ids)
# print(decoded_string)

#### another example
sequence = "I love HuggingFace"

one_tokens = tokenizer.tokenize(sequence)
one_ids = tokenizer.convert_tokens_to_ids(one_tokens)
print(one_ids)
print(len(one_ids))
# # input_ids = torch.tensor(ids)
# # print(input_ids)
# # more_dim_input_ids = torch.tensor([ids])

# print(more_dim_input_ids)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# output = model(more_dim_input_ids)
# print("Output:\n",output)

# #creating a batch
# batch = [ids, ids]
# new_input = torch.tensor(batch)
# new_output = model(new_input)
# print("Batched Output:\n", new_output.logits[0])

##another sentence
two_seq = "I hate HuggingFace so much"
two_tokens = tokenizer.tokenize(two_seq)
two_ids = tokenizer.convert_tokens_to_ids(two_tokens)

# #add cls and sep
one_ids += [tokenizer.pad_token_id] * 2
# one_ids = [tokenizer.cls_token_id] + one_ids + [tokenizer.sep_token_id]
# two_ids = [tokenizer.cls_token_id] + two_ids + [tokenizer.sep_token_id]

batched_ids = [one_ids, two_ids]
#create attention mask

one_mask = []
two_mask = []
if len(one_ids) > len(two_ids):
    one_mask = [1] * (len(one_ids)+2)
    mask_length = len(one_ids)
    two_mask = [1] * (len(two_ids)+2) + [0] * (mask_length - len(two_ids))
else:
    one_mask = [1] * len(one_ids) + [0] * (len(two_ids) - len(one_ids))
    two_mask = [1] * len(two_ids)

print(one_mask)
print(two_mask)

attention_mask = [one_mask, two_mask]

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
final_outputs = model(torch.tensor(batched_ids),attention_mask = torch.tensor(attention_mask))
                      
print(final_outputs.logits)