"""
Last Updated Date: 2024-07-05
Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
Status: Active

Type: Data Extraction Code
Objective:
1) Compiling acquired datasets from Federal Justice Center (FJC)'s Integrated Database (IDB)
2) Extracting relevant features from the compiled datasets

Associated Files:
- datasets/tab_text

Helpful Links:
- https://www.fjc.gov/research/idb/criminal-defendants-filed-and-terminated-sy-1970-through-fy-1995
"""


import pandas as pd

def open_tab_text_dataset(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path, delimiter='\t')

    # Display basic information about the dataset
    print("Columns in the DataFrame:")
    print(df.columns)

    print("\nFirst few rows of the DataFrame:")
    print(df.head())

    total_rows = len(df)
    print("Total number of rows:", total_rows) 
    return df

    
#for criminal_1970_1995.txt, total rows = 142,0711

if __name__ == "__main__":
    file_path = '/Users/turtle/Projects_Python/KVAI/data/CA/tab_text/cr70to95.txt'  # Adjust the file path as per your directory structure
    df = open_tab_text_dataset(file_path)

