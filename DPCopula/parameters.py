"""
parameters.py


Provides useful information on each dataset.
"""

from dataclasses import dataclass

input_data_dir = 'data/input/'
output_data_dir = 'data/test_output/'
aux_data_dir = 'data/auxiliary/'


@dataclass
class DataParameters:
    input_data_file: str
    attribute_file: str
    output_data_file: str


adult_params = DataParameters(
        input_data_file=input_data_dir + 'adult_data.csv',
        attribute_file=input_data_dir + 'adult_columns.csv',
        output_data_file=output_data_dir + 'adult_synthetic_data.csv',
)
