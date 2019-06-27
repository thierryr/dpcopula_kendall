"""
test_synthetic.py


Various tests to ensure that the DPCopula methods of generating synthetic data works properly.
"""

import unittest
from DPCopula.synthetic import *
from DPCopula.parameters import adult_params as params
from DPCopula.Database import Database

class TestDPCopulaMLE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = Database()
        cls.db.load_from_file(params.input_data_file, params.attribute_file)

    def test_kendall_algorithm(self):
        synthetic = kendall_algorithm(self.db, 1.0)
