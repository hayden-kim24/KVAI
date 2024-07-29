#!/usr/bin/env python3

import unittest
from sample_generator import Sample_data

class Test_Sample_data(unittest.TestCase):

    def test_initialization(self):
        sample_size = 1000
        csv_path = "/Users/turtle/Projects_Python/KVAI/data/current/csv/actual_data_India.csv"
        sample_data = Sample_data(sample_size, csv_path)
        expected = ['Dataset Note:', 'Case Type', 'Filing Number', 'Filing Date', 'Registration Number', 
                                'Registration Date', 'CNR Numbr', 'View QR Code / Cause Title', 'First Hearing Date', 
                                'Next Hearing Date', 'Case Stage', 'Nature of Disposal', 'Court Number and Judge', 
                                'Petitioner and Advocate', 'Respondent and Advocate', 'Acts', 'Under Act(s)', 
                                'Under Section(s)', 'FIR Detail', 'Police Station', 'FIR Number', 'Year', 
                                'Case History', 'Judge', 'Business on Date', 'Hearing Date', 'Purpose of Hearing', 
                                'Judge', 'Business on Date', 'Hearing Date', 'Purpose of Hearing', 'Judge', 
                                'Business on Date', 'Hearing Date', 'Purpose of Hearing', 'Judge', 'Business on Date', 
                                'Hearing Date', 'Purpose of Hearing', 'Interm Orders', 'Order Number', 'Order Date', 
                                'Order Details', 'Order Number', 'Order Date', 'Order Details']
        return self.assertEqual(sample_data.data_fields, expected)

if __name__ == '__main__':
    unittest.main()