# Learning Notes


# Learning Note: Date & Number
Learning Note #12: 2024-07-17
## Learning Topic
dim=-1 for torch.nn.Functional.softmax(outputs.logits, dim =-1)
# Description
the last dimension -> applies softmax

example: if the logits have the shape 2 x 3:
[1.3043, 3.343, 2342.24324],
[0.234, 0.234, 0.234]

Then the last dimension here would be 3
and softmax will be applied for each of the 3 values within an example
thereby giving us the probabilities of each of the classes.
rlly cool!

## --------- DONE---------


# Learning Note: Date & Number
Learning Note #11: 2024-07-17
## Learning Topic
Single Asterik * to unpack any iterable that Python provides
Double Asterik ** to unpack dictionary inputs
## Relevant File & Code
File Name: transformers/2_tensors.py

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs) # inputs will have input_ids and attention_mask like the following:
{'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
             0,     0,     0,     0,     0,     0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

## What I learned
putting ** in front of a variable will unpack the dictionary.
If not, the variable "inputs" will be treated as a single object, like a tuple.
Also, "The single asterisk operator * can be used on any iterable that Python provides, while the double asterisk operator ** can only be used on dictionaries." https://realpython.com/python-kwargs-and-args/

## Additional Feature
my_first_list = [1, 2, 3]
my_second_list = [4, 5, 6]
my_merged_list = [*my_first_list, *my_second_list]

print(my_merged_list) # my_merged_list will be [1, 2, 3, 4, 5, 6]
-> This is rlly cool!

a = [*"RealPython"]
print(a) # a will be ['R', 'e', 'a', 'l', 'P', 'y', 't', 'h', 'o', 'n']

## --------- DONE---------

# Learning Note: Date & Number
Learning Note #10: 2024-07-17
## Learning Materials
HuggingFace NLP Course Chapter 1, https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt
# Description
Learned the terminology for fine-tuning pre-trained model
-> "TRANSFER LEARNING"

## --------- DONE---------

# Learning Note: Date & Number
Learning Note #9: 2024-07-16
## Learning Topic
Transfer Learning
## Learning Materials
HuggingFace NLP Course Chapter 1, https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt
# Description
Learned the terminology for fine-tuning pre-trained model
## --------- DONE---------

# Learning Note: Date & Number
Learning Note #8: 2024-07-16
## Learning Topic
Transformers Library: Pipeline function
## Learning Materials
HuggingFace NLP Course Chapter 1, https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt
# Description
Learned how to deploy a pre-trained LLM to serve a specific function using pipeline function.

!!! Zero-shot-classification seems to be a PERFECT way to figure out the intent of the user! SO COOL!! YIPPIE!!

?? Maybe use named entity recognition for user inqueries?? Like the ones that ask for case information and stuff?? ex. "Do you have any information about John Doe?" (person) v. "Do you have any information about NYPD?" (organization)
-> ??Maybe have a separate tagged files for list of police stations & list of 
Indian cities?? 
Follow-up:
!!! It recognized Ramgopalpet as a location!! So I guess for police station & states, I should consider name entity recognition!!! 

??question_answering is not functioning as I want it to be. Not giving me good scores like zero shot classification one.??
-> Maybe follow up later on.

??Summarizer might be useful for summarizing the case info to the users???
-> Maybe consider it.

!!Translator feature -> use it for Hindi"


Other modes for pipeline() is cool too.
Learned about Text generation, mask filling, etc.
# Related File
"self_training/transformers/1_pipeline.py"

## --------- DONE---------

# Learning Note: Date & Number
Learning Note #7: 2024-07-16
## Learning Topic
Segmentation Fault for Importing Libraries in Python
## Related Error
Error #3: 2024-07-16
# Description
Segmentation Fault -> Memory issue.
Occurs when we try to access memory location that is not available 
(either memory is not allocated OR protected by the operating system).

APPARENTLY certain packages importing might lead to segmentation fault.
Interesting. The Number 11 is just for MacOs, no specific meaning.

## --------- DONE---------

# Learning Note: Date & Number
Learning Note #6: 2024-07-15
## Learning Topic
New Pakcage -- Faker: creating string with Numbers and letters
## How-to
https://faker.readthedocs.io/en/master/providers/baseprovider.html
ex: faker.bothify(text='????############', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')e


## --------- DONE---------



# Learning Note: Date & Number
Learning Note #5: 2024-07-15
## Learning Topic
Python Infrastructure -- Importing Files from Different Folders
## Error I was getting
When running Python interpreter could not locate SampleData class from sample_generator.py bec


## --------- DONE---------



# Learning Note: Date & Number
Learning Note #4: 2024-07-15
## Learning Topic
New Python Package -- Unittest
## What I learned
Can use unittest package to do testing for my functions. 
Created test_sample_generator.py file based on it
## Another thing I learned
test methods MUST start with the prefix "test_". If not, the test will not run


## --------- DONE---------


# Learning Note: Date & Number
Learning Note #3: 2024-07-15

## Learning Topic
Code Debugging -- List

## Associated Files & Functions
"sample_generator.py": choose_data_fields function for the Sample_data class

## Previous Code
    for each_idx in choices_indices:
        print(self.data_fields[each_idx])
        self.data_fields.remove(self.data_fields[each_idx])
## Issue with the Previous Code
 I wanted to remove "Dataset Notes" (initial idx 0) and "View QR Code" (initial idx 7) from the data fields list.
 But since it removed Dataset Notes AND THEN removed idx 7, the code eventually removed "First Hearing Date" (initial idx 6) instead of "View QR Code" (initial idx 7).

 ## Solution
Creating a separate list of data fields that I want to remove BEFORE I start removing them.

 ## New Code

items_remove = []
for each_idx in choices_indices:
    items_remove.append(self.data_fields[each_idx])

for each_item in items_remove:
    self.data_fields.remove(each_item)

-> Works like a charm!
## --------- DONE---------


# Learning Note: Date & Number
Learning Note #2: 2024-07-15
## Learning Topic
Python Infrastracture -- interpretation order for Python interpreter
## Associated Files & Functions
"sample_generator.py": __init__ and add_data_fields for the Sample_data class
## Question I had
whether I have to declare add_data_fields beofre __init__ if I wanted to use that function in the __init__ function
## Answer
It is perfectly fine to declare the add_data_fields method after the __init__ method. 
In Python, one can declare methods in a class in any order.
Why? -> The Python interpreter processes the entire class definition before instantiating objects, so the order of method definitions does not affect the ability to call them within the class.
## --------- DONE---------

# Learning Note: Date & Number
Learning Note #1: 2024-07-10
## Learning Topic
New Python Package -- Faker
## What I learned
Faker package for generating "fake" data. 
https://github.com/joke2k/faker
## --------- DONE---------




# Error Logs

# Error Date & Number
Error #6: 2024-07-18
## Error Status
RESOLVED 
## Related Error
Error #3
## Error Type
"Segmentation Fault: 11" for "from datasets import load_dataset"

## Error Description
Terminal keep giving me "Segmentation Fault: 11" message even though I already commented out  all the code except for the import statements.

## Solution
This took an hour and a half to resolve.
Somehow running  conda install -c huggingface -c conda-forge datasets 
over and over worked.
I don't understand why it didn't work at first.
Felt pretty burned out after spending an hour and a half on this.
But I'm proud that it works now & that I can run datasets module on my laptop. Yippie? 

## --------- RESOLVED ---------






# Error Date & Number
Error #5: 2024-07-17
## Error Status
RESOLVED
## Related Error
N/A
## Error Type
Pytorch: Tensor size Issue + List v. Tensor
## Error Description 
to convert a list to a tensor, must apply two brakets
so for instance, if we have the attention mask of shape [1,0,0,0,0]
we must declare a tensor like "tensor = torch.tensor([attention_mask])

also make sure that tensor size for input ids match the size for attention mask -- or else there will be errors.

## --------- RESOLVED ---------



# Error Date & Number
Error #4: 2024-07-17
## Error Status
RESOLVED
## Related Error
N/A
## Error Type
Tensorflow installation error - requirement not matching even tho I did check the python version
## Solution Reference
https://stackoverflow.com/questions/48720833/could-not-find-a-version-that-satisfies-the-requirement-tensorflow
## Solution
pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl

## --------- RESOLVED ---------


# Error Date & Number
Error #3: 2024-07-16
## Error Status
RESOLVED -- See Error #6
## Related Error
N/A
## Error Type
"Segmentation Fault: 11"

## Error Description
Terminal keep giving me "Segmentation Fault: 11" message even though I already commented out  all the code except for the import statements.

## Solution
Isolated each line from the very beginning. 

import json -> works fine
from datasets import Dataset -> "Segmentation Fault:11" message

## --------- RESOLVED ---------


# Error Date & Number
Error #2: 2024-07-10
## Error Status
RESOLVED
## Related Error
Error #1 - 2024-07-05
## Error Type
File Compression + Decompression Issue -- File Content Deleted for data/tab_text/criminal_1970_1995.txt
## Error Description
The file content gets deleted after I decompress the file
## Solution
Keep the file in the local directory. Add it to gitignore so that it won't get added to git. 
## --------- RESOLVED ---------

# Error Date & Number
Error #1: 2024-07-05
## Error Status
RESOLVED
## Error Type
Git Issue -- Large Files
## Error Summary
Git push not woring
## Error Description
"file data/tab_text/criminal_1970_1995.txt:169.98 MB" -> file too big
## Error Message
Enumerating objects: 17, done.
Counting objects: 100% (17/17), done.
Delta compression using up to 8 threads
Compressing objects: 100% (13/13), done.
Writing objects: 100% (15/15), 19.51 MiB | 2.68 MiB/s, done.
Total 15 (delta 0), reused 0 (delta 0), pack-reused 0
remote: error: Trace: 0aed480417aa440d9e419dc5c6e233c00efc71190161bc1bb9dfea6fe91c6ba4
remote: error: See https://gh.io/lfs for more information.
remote: error: File data/tab_text/criminal_1970_1995.txt is 169.98 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
To https://github.com/hayden-kim24/KVAI.git
 ! [remote rejected] main -> main (pre-receive hook declined)
error: failed to push some refs to 'https://github.com/hayden-kim24/KVAI.git'
## Attempted Solution: 
brew install git-lfs
## Follow up Error #1
brew install not working
## Follow up Solution #1:
git lfs migrate info
git lfs migrate import --include="*.txt"
* reference #1: https://github.blog/2017-06-27-git-lfs-2-2-0-released/
* reference #2: https://stackoverflow.com/questions/33330771/git-lfs-this-exceeds-githubs-file-size-limit-of-100-00-mb
## Follow up Error #2:
"batch response: This repository is over its data quota. Account responsible for LFS bandwidth should purchase more data packs to restore access."
## Follow Up Solution #2:
gzip "data/tab_text/criminal_1970_1995.txt"
## --------- RESOLVED ---------




