# Learning Notes


# Learning Note: Date & Number
Learning Note #8: 2024-07-16
## Learning Topic
Transformers Library: Pipeline function
# Description
Learned how to deploy a pre-trained LLM to serve a specific function using pipeline function.
Zero-shot-classification seems to be a PERFECT way to figure out the intent of the user! SO COOL!! YIPPIE!!
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
Error #3: 2024-07-16
## Error Status
IN PROGRESS
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




