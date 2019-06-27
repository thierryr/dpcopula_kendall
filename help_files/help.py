"""
help.py

Functionality to show print help for python scripts.

Generates a help function to display the contents of the relevant help
file for a script.
"""


def generate_help_function(module):
    def print_help():
        help_file = f'help_files/{module}_help.txt'
        with open(help_file, 'r') as file:
            for line in file:
                print(line, end='')

    return print_help
