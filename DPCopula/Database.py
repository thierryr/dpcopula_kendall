"""
Database.py


Defines a database object to hold the input data with useful related
functions.
"""

import numpy as np


class Database:
    def __init__(self, data=None, attr_table=None):
        """
        Container class for a database.

        `data` is a 2-d array containing rows of data, with the entries
        in each row corresponding to values in `attr_table`, which is a
        dict whose key-value pairs are attributes and their possible
        values.

        Also generates useful data such as a one-way histogram and
        performs a numerical data conversion. This conversion will
        assign arbitrary numbers to non-ordinal data.

        Data and attributes can be provided directly through `__init__`
        or from file through `load_from_file`.
        """
        self.data = data
        self.attr_table = attr_table

        if data is not None and attr_table is not None:
            self.process_data()

    def load_from_file(self, source_file, attr_file):
        """
        Takes in a file containing the data as well as the data
        attributes (fields).  Reads the data from the file and generates
        a list of entries which themselves contain the attribute values
        of each row.
        """

        self.data = self.get_data_from_file(source_file)
        self.attr_table = self.get_attrs_from_file(attr_file)

        self.process_data()

    def save_to_file(self, dest_file):
        with open(dest_file, 'w') as file:
            for row in self.data:
                file.write(','.join(map(str, row)) + ',\n')

    def process_data(self):
        """Computes useful information based off the raw data."""
        self.numerical_data = self.convert_data()
        self.num_attrs = len(self.attr_table.keys())
        self.num_rows = len(self.data)
        self.one_way_histogram = self.generate_one_way_hist()
        self.numerical_attr_table = {}
        for attr, vals in self.attr_table.items():
            numerical_vals = [self.get_numerical_value(attr, val)
                              for val in vals]
            self.numerical_attr_table[attr] = numerical_vals

    def get_marginal_histogram(self, attr):
        """
        Calculate the marginal histogram for the specified attribute.

        Returns a list of tuples of the form (value, count) such that
        `value` occurs `count` times in the specified margin.
        """

        marginal = self.numerical_data[:, attr]
        values = self.numerical_attr_table[attr]
        histogram = [[key, 0] for key in sorted(values)]

        for entry in marginal:
            val_index = values.index(entry)
            histogram[val_index][1] += 1

        return histogram

        # marginal = self.data[:, attr]
        # histogram = [0] * len(values)
        # hist_dict = Counter(marginal)
        # histogram = [(k, hist_dict[k]) for k in sorted(hist_dict.keys())]

        # return histogram

    def get_numerical_value(self, attribute, value):
        """
        Performs a numerical conversion of a piece of data.

        If the data is numerical, return its numerical value, otherwise
        return its position in the list of attribute values.
        """

        try:
            numerical = int(value)
        except ValueError:
            numerical = int(self.attr_table[attribute].index(value))

        return numerical

    def get_attribute_value(self, attribute, num_val):
        """Convert numerical data back to its original value."""
        values = self.attr_table[attribute]
        try:
            int(values[0])
        except ValueError:
            return values[int(num_val)]

        return str(int(num_val))

    def convert_data(self):
        """Converts the database to numerical form."""
        num_data = [[self.get_numerical_value(attr, val)
                     for attr, val in enumerate(line)] for line in self.data]

        return np.array(num_data)

    def generate_one_way_hist(self):
        """
        Generate a one way histogram of each attribute value.

        Returns a dict of the counts with the key containing the attr
        and its value in the string [attr]_[val].
        """

        hist = {f'{attr}_{val}': 0 for attr in self.attr_table
                for val in self.attr_table[attr]}

        for entry in self.data:
            for attr, val in enumerate(entry):
                key = f'{attr}_{val}'
                hist[key] += 1

        return hist

    def generate_two_way_hist(self):
        attr_val_pairs = [(attr, val) for attr in self.attr_table
                          for val in self.attr_table[attr]]

        combinations = [(avp1, avp2) for avp1 in attr_val_pairs
                        for avp2 in attr_val_pairs if avp1[0] < avp2[0]]

        matches = []
        for comb in combinations:
            # print(comb)
            count = 0
            for line in self.data:
                avp1, avp2 = comb
                if avp1[1] == line[avp1[0]] and avp2[1] == line[avp2[0]]:
                    count += 1

            matches.append(count)

        return list(zip(combinations, matches))

    @staticmethod
    def get_attrs_from_file(attr_file):
        """
        Load the atrributes and their values from file.

        Creates a dict with each key-value pair being the attribute and
        corresponding list of possible values.
        """

        attr_table = {}

        with open(attr_file, 'r') as file:
            attr_pairs = [(line.strip().split('_'))
                          for line in file.readlines()]

        for (attr, value) in attr_pairs:
            attr = int(attr)
            if attr in attr_table:
                attr_table[attr].append(value)
            else:
                attr_table[attr] = [value]

        return attr_table

    @staticmethod
    def get_data_from_file(data_file):
        """
        Load the data from file.

        Creates a list of list with each inner list corresponding to one
        data entry.
        """
        data = np.genfromtxt(data_file, delimiter=',', dtype=str)
        return data[:, :-1]  # Database contains comma before newline

        # with open(data_file, 'r') as file:
        #     data = file.readlines()

        # data = [line[:-2].split(',') for line in data]

        # return data

    @staticmethod
    def avp_to_key(attr_val_pair):
        return f'{attr_val_pair[0]}_{attr_val_pair[1]}'

    @staticmethod
    def key_to_avp(key):
        parts = key.split('_')
        return (int(parts[0]), parts[1])
