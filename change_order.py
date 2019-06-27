"""
change_order.py


Create a new attributes file with a randomised order.
Also creates a file containing the indices of the old and new binary
columns where the position of the number indicates the old column and
the number indicates the new column.

E.g. 2,0,1 would indicate the following mappings:
    col 0 -> col 2
    col 1 -> col 0
    col 2 -> col 1
"""

from help_files.help import generate_help_function

import random
import os
import sys

print_help = generate_help_function('change_order')


if __name__ == '__main__':
    try:
        set_num = sys.argv[1]
    except IndexError:
        print_help()
        quit()

    if set_num == 'help':
        print_help()

    else:
        with open('data/input/adult_columns.csv', 'r') as file:
            attr_list = [attr.strip() for attr in file.readlines()]

        num_attrs = 14

        # Group attribute values
        # All values from the same column will be collected into a list
        grouped_attrs = [[] for _ in range(num_attrs)]
        for attr in attr_list:
            attr_id = int(attr.split('_')[0])
            grouped_attrs[attr_id].append(attr)

        # Randomise each group separately as to preseve the attribute
        # order (attr 0 comes before attr 1 etc.)
        for vals in grouped_attrs:
            random.shuffle(vals)

        # Flatten the randomised list
        random_order = [val for attr in grouped_attrs for val in attr]

        # Obtain the column mappings for each attribute value
        mappings = []
        for i in range(len(attr_list)):
            orig = attr_list[i]
            random = random_order[i]

            mappings.append(str(random_order.index(orig)))

        # Save output to file
        folder = f'data/experiment_data/attribute_sets/set_{set_num}'
        os.makedirs(folder, exist_ok=True)

        with open(f'{folder}/adult_columns.csv', 'w+') as file:
            for val in random_order[:-1]:
                file.write(f'{val}\n')
            file.write(random_order[-1])  # No trailing line in file

        with open(f'{folder}/column_order.csv', 'w+') as file:
            file.write(','.join(mappings))
