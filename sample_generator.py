# Last Updated Date: 2024-07-15
# Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
# Status: Active / In Development
# Used Codeium's AI to generate comments for this file.

import csv
import json
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from typing import List 
from typing import Optional

class Sample_data:
    def __init__(self, sample_size: int, csv_path: Optional[str] = None) -> None:
        """
        Initializes the Sample_data class with the given sample size.

        Parameters:
            sample_size (int): The size of the sample data to be generated.

        Returns:
            None

        Initializes:
            self.data_fields (list): A list of data fields extracted from the CSV file.
            self.sample_size (int): The size of the sample data to be generated.
        """

        self.data_fields = {}
        if csv_path:
            self.add_data_fields(csv_path)

        self.sample_size = sample_size

    def add_data_fields(self, csv_path: str) -> None:
        """
        Lists the initial data fields extracted from the CSV file.

        This function reads the CSV file located at "/Users/turtle/Projects_Python/KVAI/data/current/csv/actual_data_India.csv"
        
        and extracts the data fields from the first column of each row. The extracted data fields are stored in the
        `self.data_fields` dictionary with the index of the row as the key.

        Parameters:
            - csv_path (str): The path to the CSV file that has all the data fields & the actual data.

        Returns:
            None

        Updates:

        Prints:
            - "Current Data Fields"
            - The `self.data_fields` dictionary containing the extracted data fields.

        Note:
            - The CSV file is assumed to have a header row.
            - The function assumes that the first column of each row contains the data field.
            - The function does not handle any exceptions that may occur during the file reading or data extraction process.
        """
        sampler = Faker()

        # Extract data fields from csv file
        csv_file = csv_path

        #reset data fields
        self.data_fields = {} 

        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row[0]) > 0: #skip empty data fields
                    self.data_fields[(row[0].strip())] = row[3].strip()
        print ("\n Current Data Fields (Before Cleaning):")
        for idx, each_field in enumerate(self.data_fields):
            print(idx, each_field)

    def write_sql_schema(self,sql_script_file:str) -> None:

        with open(sql_script_file, 'w') as sql_file:
        # Write SQL script header
            sql_file.write('-- SQL script generated from CSV\n\n')
            sql_file.write('-- Create table to store data fields\n')
            sql_file.write('CREATE TABLE data_fields (\n')
            sql_file.write('    id INT AUTO_INCREMENT PRIMARY KEY,\n')
            sql_file.write('    field_name VARCHAR(255) NOT NULL,\n')
            sql_file.write('    data_type VARCHAR(50) NOT NULL\n')
            sql_file.write(');\n\n')
            for each_key in self.data_fields:
                sql_file.write(f'INSERT INTO data_fields (field_name, data_type) VALUES ("{each_key}", "{self.data_fields[each_key]}");\n')

        print(f"SQL script generated and saved to '{sql_script_file}'")

    def choose_data_fields(self,rmv_data_fields:List[str]) -> None:

        print(f"\nRemoving Data Fields:{rmv_data_fields}")

        for each_item in rmv_data_fields:
            if each_item in self.data_fields.keys():
                del self.data_fields[each_item]

        print("\nUpdated Data Fields")
        for idx, each_field in enumerate(self.data_fields):
            print(idx, each_field)

    def summarize_datafields(self) -> None:
        print("\nSummary of the Current Data Fields:")
        print(f"Total Number of Data Fields: {len(self.data_fields)}")

    def generate_date(self):
        start_date = datetime(1900, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_data = pd.Series([start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(self.sample_size)])
        print(date_data)

# # Generate numeric data
# numeric_data = np.random.uniform(0, 100, sample_size)

# # Generate categorical data
# categories = ['Category A', 'Category B', 'Category C']
# probabilities = [0.4, 0.3, 0.3]
# categorical_data = pd.Series(np.random.choice(categories, sample_size, p=probabilities))

# # Generate text data
# text_data = [fake.text() for _ in range(sample_size)]

# # Generate date data
# start_date = datetime(2023, 1, 1)
# end_date = datetime(2023, 12, 31)
# date_data = pd.Series([start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(sample_size)])

# # Generate spatial data
# spatial_data = gpd.GeoSeries([Point(np.random.uniform(min_lon, max_lon), np.random.uniform(min_lat, max_lat)) for _ in range(sample_size)])

# # Create a dictionary to store all generated data
# sample_data = {
#     'NumericField': list(numeric_data),
#     'CategoricalField': list(categorical_data),
#     'TextField': text_data,
#     'DateField': [date.strftime('%Y-%m-%d') for date in date_data],
#     'SpatialField': [{'longitude': point.x, 'latitude': point.y} for point in spatial_data]
# }

# # Save sample data to a JSON file
# with open('sample_data.json', 'w') as f:
#     json.dump(sample_data, f, indent=4)

# print("Sample data has been generated and saved to 'sample_data.json'.")

if __name__ == "__main__":
    
    sample_size = 1000
    csv_path = "/Users/turtle/Projects_Python/KVAI/data/current/csv/actual_data_India.csv"

    data = Sample_data(sample_size, csv_path)
    print(data.data_fields)
    exclude_fields = ["Dataset Note:","View QR Code / Cause Title"] #Corresponds to "Header" & "View QR Code" Button
    data.choose_data_fields(exclude_fields)
    sql_script_file = "sample_schema.sql"
    data.write_sql_schema(sql_script_file)
    #data.generate_date()




