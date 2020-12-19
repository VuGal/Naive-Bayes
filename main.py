#!/usr/bin/env python3

# This file contains the main script of the project

import sys
import subprocess
from src.iris import Iris
from src.pima_indians_diabetes import PimaIndiansDiabetes


def main():

    print('\n==============================================')
    print('Welcome to the Naive Bayes Classifier project!')
    print('==============================================')

    returned_from_function = True

    while True:

        if returned_from_function == True:
            print('\nChoose the action:')
            print('\n1. Run the Naive Bayes implementation using Iris dataset.')
            print('2. Run the Naive Bayes implementation using Pima Indians Diabetes dataset.')
            print('3. Run unit tests.')
            print('4. Run acceptance tests.')
            print('5. Show this algorithm vs. scikit-learn comparison.')
            print('6. Exit the program.\n')

        returned_from_function = False

        try:
            choice = input('Choice: ')
        except KeyboardInterrupt:
            print('\n\nGoodbye!\n')
            sys.exit()

        if choice not in ['1', '2', '3', '4', '5', '6']:
            print('Wrong choice! Please choose option 1-6.')

        elif choice == '1':

            try:
                iris = Iris()
                iris.run()
                returned_from_function = True
                continue
            except KeyboardInterrupt:
                returned_from_function = True
                continue

        elif choice == '2':

            try:
                pid = PimaIndiansDiabetes()
                pid.run()
                returned_from_function = True
                continue
            except KeyboardInterrupt:
                returned_from_function = True
                continue

        elif choice == '3':

            try:
                output = subprocess.run(f'py.test tests/unit_tests', shell=True,
                                        stdout=subprocess.PIPE, universal_newlines=True).stdout

                print(f'{output}')
                returned_from_function = True
                continue
            except KeyboardInterrupt:
                returned_from_function = True
                continue

        elif choice == '4':

            try:
                output = subprocess.run(f'py.test tests/acceptance_tests', shell=True,
                                        stdout=subprocess.PIPE, universal_newlines=True).stdout
                print(f'{output}')
                returned_from_function = True
                continue
            except KeyboardInterrupt:
                returned_from_function = True
                continue

        elif choice == '5':

            try:
                print('\n**************************************')
                print('           iris.csv dataset')
                print('**************************************')
                output = subprocess.run(f'python3 tests/acceptance_tests/test_iris_scikit_learn_comparison.py',
                                        shell=True, stdout=subprocess.PIPE, universal_newlines=True).stdout
                print(f'{output}')
                print('\n\n\n**************************************')
                print('  pima-indians-diabetes.csv dataset')
                print('**************************************')
                output = subprocess.run(f'python3 \
                                        tests/acceptance_tests/test_pima_indians_diabetes_scikit_learn_comparison.py',
                                        shell=True, stdout=subprocess.PIPE, universal_newlines=True).stdout
                print(f'{output}')
                returned_from_function = True
                continue
            except KeyboardInterrupt:
                returned_from_function = True
                continue

        elif choice == '6':

            print('\n\nGoodbye!\n')
            sys.exit()

        else:
            raise


if (__name__ == '__main__'):

    main()

