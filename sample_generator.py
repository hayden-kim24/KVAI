# Last Updated Date: 2024-07-10
# Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
# Status: Active
# Used Codeium's AI to generate comments for this file.

import csv
import json
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from typing import List 

class Sample_data:
    def __init__(self, sample_size: int):
        """
        Initializes the Sample_data class with the given sample size.

        Parameters:
            sample_size (int): The size of the sample data to be generated.

        Returns:
            None
        """
        self.data_fields = []
        self.sample_size = sample_size

    def list_data_fields(self, csv_path: str) -> None:
        """
        Lists the initial data fields extracted from the CSV file.

        This function reads the CSV file located at "/Users/turtle/Projects_Python/KVAI/data/current/csv/actual_data_India.csv"
        
        and extracts the data fields from the first column of each row. The extracted data fields are stored in the
        `self.data_fields` dictionary with the index of the row as the key.

        Parameters:
            None

        Returns:
            None

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
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row[0]) > 0: #skip empty data fields
                    self.data_fields.append(row[0])
        print ("Current Data Fields")
        for idx, each_field in enumerate(self.data_fields):
            print(idx, each_field)

    def choose_data_fields(self,choices_indices:List[int]) -> None:

        print("\nWe will choose the following data fields to generate:")
        for each_idx in choices_indices:
            print(self.choices_indices[each_idx])
            self.data_fields.remove(self.choices_indices[each_idx])
        print("\nUpdated Data Fields")
        for idx, each_field in enumerate(self.data_fields):
            print(idx, each_field)

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
    
    data = Sample_data(1000)

    data.list_data_fields("/Users/turtle/Projects_Python/KVAI/data/current/csv/actual_data_India.csv")

    exclude_indices = [0,7] #Corresponds to "Header" & "View QR Code" Button
    # data.exclude_data_fields(exclude_indices)
    data.generate_date()




