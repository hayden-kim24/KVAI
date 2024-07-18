# Last Updated Date: 2024-07-18
# Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
# Purpose: Self-Training (Not to be used for actual app)
# Status: In Progres
# Based on: Hugging Face NLP Course Chapter 3,https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt
# Code is directly from Hugging Face NLP Course website

import sys
egg_path = '__MODULE_PATH__/datasets-0.0.9-py3.5.egg'
# sys.path.append(egg_path)
# import os
# import numpy
# os.environ['OPENBLAS_NUM_THREADS'] = '5'

# import sys  
# sys.path.append(r"D:/Applications/miniconda3/lib/python3.12/site-packages"); 
from datasets import load_dataset


raw_ds = load_dataset("glue","mrpc")

# raw_ds = load_dataset("glue","mrpc")
# print(raw_ds)

