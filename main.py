#!/usr/bin/env python3

# This file will contain the main script of the project

import sys
import subprocess


def main():

    print('\n==============================================')
    print('Welcome to the Naive Bayes Classifier project!')
    print('==============================================')

    print('\nChoose the action:')
    print('\n1. Run the Naive Bayes implementation using Iris dataset.')
    print('2. Run the Naive Bayes implementation using Pima Indians Diabetes dataset.')
    print('3. Run unit tests')
    print('4. Run acceptance tests\n')


    while True:

        try:
            choice = input('Choice: ')
        except KeyboardInterrupt:
            print('Goodbye!')
            sys.exit()

        if choice not in ['1', '2', '3', '4']:
            print('Wrong choice! Please choose option 1-4.')

        elif choice == '1':

            try:
                print('Iris NB implementation.')
                break
            except KeyboardInterrupt:
                break

        elif choice == '2':

            try:
                print('PID NB implementation.')
                break
            except KeyboardInterrupt:
                break

        elif choice == '3':

            try:
                output = subprocess.run(f'py.test tests/unit_tests', shell=True, stdout=subprocess.PIPE, universal_newlines=True).stdout
                print(f'{output}')
                break
            except KeyboardInterrupt:
                break

        elif choice == '4':

            try:
                output = subprocess.run(f'py.test tests/acceptance_tests', shell=True, stdout=subprocess.PIPE, universal_newlines=True).stdout
                print(f'{output}') 
                break
            except KeyboardInterrupt:
                break

        else:
            raise


if (__name__ == '__main__'):

    main()

