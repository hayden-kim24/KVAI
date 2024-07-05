"""
Last Updated Date: 2024-07-05
Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
Status: Deprecated
- No longer using this file! Turns out that the compiled dataset is available as an tab-delimited text file.

Type: Supplementary Exploration Code
Objective: Exploring SAS datasets available in Federal Justice Center (FJC)'s Integrated Database (IDB)
Associated Files:
- datasets/sas/1995_criminal_case.sas7bdat
- datasets/sas/1970_civil_case.sas7bdat

Helpful Links:
- https://www.fjc.gov/research/idb/criminal-defendants-filed-and-terminated-sy-1970-through-fy-1995
"""

import pandas as pd
from sas7bdat import SAS7BDAT

sas_file = 'datasets/sas/1995_criminal_case.sas7bdat'

with SAS7BDAT(sas_file) as f:
    sas_data = f.to_data_frame()

# Display basic information about the dataset
print("Columns in SAS dataset:")
print(sas_data.columns)
print("\nFirst few rows of SAS dataset:")
print(sas_data.head())