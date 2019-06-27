"""
test_Database.py


Various tests to ensure the Database class operates correctly
"""

import unittest
from DPCopula.Database import Database
from DPCopula.parameters import adult_params as params
from DPCopula.synthetic import get_dp_marginals


class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = Database()
        cls.db.load_from_file(params.input_data_file, params.attribute_file)

    def test_attr_retrieval(self):
        self.assertEqual(list(self.db.attr_table.keys()), list(range(14)))
        expected_num_values = [9, 9, 16, 16, 7, 15, 6, 5, 2, 23, 32, 10, 42, 2]
        for index, expected in enumerate(expected_num_values):
            self.assertEqual(len(self.db.attr_table[index]), expected)

    def test_one_way_histogram(self):
        expected_histogram = {}
        with open('data/auxiliary/adult_oneway.csv', 'r') as file:
            for line in file:
                attr, count = line.strip().split(',')
                expected_histogram[attr] = int(count)

        for key, count in self.db.one_way_histogram.items():
            self.assertEqual(count, expected_histogram[key])

    def test_get_numerical_value(self):
        self.assertEqual(self.db.get_numerical_value(1, 'Never-worked'), 3.0)
        self.assertEqual(self.db.get_numerical_value(3, '16'), 16.0)

    def test_database_properties(self):
        self.assertEqual(self.db.numerical_data.shape, (32561, 14))

    # def test_get_marginal_histogram(self):
    #     expected_marginal = [(1.0, 1657), (2.0, 8054), (3.0, 8613),
    #                          (4.0, 7175), (5.0, 4418), (6.0, 2015),
    #                          (7.0, 508), (8.0, 78), (9.0, 43)]
    #     self.assertEqual(self.db.get_marginal_histogram(0), expected_marginal)

    def test_get_marginal_histogram_shorter(self):
        short_db = Database()
        short_db.load_from_file('data/auxiliary/short_orig.csv',
                                'data/input/adult_columns.csv')
        print(short_db.data)
        print(short_db.numerical_attr_table)

        marginal = short_db.get_marginal_histogram(0)
        print(marginal)
        dp_marginals = get_dp_marginals(short_db, 1)
        print(dp_marginals[0])

