#!/usr/bin/env python3

# This file holds the Naive Bayes classifier implementation for 'pima-indians-diabetes.csv' dataset.

import csv
from random import seed
from naive_bayes import NaiveBayesClassifier


class PimaIndiansDiabetes:

    """

    Works on pima-indians-diabetes.csv dataset and interactively performs the following actions:\n
    1. Classify new data entered by user.\n
    2. Calculate the algorithm implementation accuracy.\n
    3. Show dataset description (pima-indians-diabetes.names file).\n
    4. Show dataset rows.

    """

    def __init__(self):

        self.dataset_filename = 'datasets/pima-indians-diabetes.csv'
        self.description_filename = 'datasets/pima-indians-diabetes.names'
        self.nbc = NaiveBayesClassifier()
        self.dataset = self.nbc.load_dataset_from_csv(self.dataset_filename)


    def data_preprocessing(self):

        """

        Converts class names (strings) to ints and class values to floats.

        Args:
            None.

        Returns:
            Nothing.

        """

        for i in range(len(self.dataset[0]) - 1):
            self.nbc.convert_class_values_to_floats(self.dataset, i)

        self.nbc.map_class_names_to_ints(self.dataset, len(self.dataset[0]) - 1, numbers_already=True)


    def classify_data(self):

        """

        Creates a new row with values inputted by the user, then classifies it to the proper class
        using Naive Bayes Classifier algorithm.

        Args:
            None.

        Returns:
            Nothing.

        """

        print('\nEnter the data to be classified.\n')

        attributes = {
            'Number of times pregnant: ' : 0.0,
            'Plasma glucose concentration a 2 hours in an oral glucose tolerance test: ' : 0.0,
            'Diastolic blood pressure (mm Hg): ' : 0.0,
            'Triceps skin fold thickness (mm): ' : 0.0,
            '2-Hour serum insulin (mu U/ml): ' : 0.0,
            'Body mass index (weight in kg/(height in m)^2): ' : 0.0,
            'Diabetes pedigree function: ' : 0.0,
            'Age (years): ' : 0.0
        }

        for attr in attributes:

            correct_input = False

            while correct_input == False:

                try:
                    attr_value = float(input(attr))
                    correct_input = True
                except:
                    print('Incorrect value! Please enter an integer or a floating point number.')

            attributes[attr] = attr_value

        print('\nEntered attributes:\n')

        for attr in attributes:
            print(f'{attr}{attributes[attr]}')

        print()

        confirm_sign = ''

        while confirm_sign not in ['y', 'Y', 'n', 'N']:
            confirm_sign = input('Confirm (y/n): ')

        if confirm_sign in ['n', 'N']:
            return

        model = self.nbc.calculate_class_parameters(self.dataset)
        label = self.nbc.predict(model, list(attributes.values()))

        # Original dataset contains class names represented as numbers,
        # so it's needed to print the labels explicitly
        if label == 0:
            print(f'\nThe entered entity was classified as: Negative')
        elif label == 1:
            print(f'\nThe entered entity was classified as: Positive')
        else:
            raise


    def calculate_accuracy(self, n_folds=5):

        """

        Calculates algorithm accuracy by using evaluate_algorithm() function.

        Args:
            n_folds (int)
                Number of folds used in the k-fold cross validation split algorithm.

        Returns:
            accuracy
                Calculated classifier accuracy in percent.

        """

        scores = self.nbc.evaluate_algorithm(self.dataset, n_folds)

        print('\n\nCalculating the accuracy of the classifier using the pima-indians-diabetes.csv dataset...')
        print('\nResampling: k-fold cross validation split')

        accuracy = (sum(scores) / float(len(scores)))
        print(f'\nAccuracy ({n_folds} folds): {round(accuracy, 3)} %\n')

        return accuracy


    def show_dataset_description(self):

        """

        Prints the 'pima-indians-diabetes.names' file to the console output.

        Args:
            None.

        Returns:
            Nothing.

        """

        with open(self.description_filename, 'r') as f:

            csv_reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for row in csv_reader:
                for word in row:
                    print(word, end='')
                print()


    def show_dataset_rows(self):

        """

        Prints the 'pima-indians-diabetes.csv' file to the console output.

        Args:
            None.

        Returns:
            Nothing.

        """

        with open(self.dataset_filename, 'r') as f:

            csv_reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for row in csv_reader:
                for i in range(len(row)-1):
                    print(row[i], end=',')
                print(row[len(row)-1])


    def run(self):

        """

        Creates the interactive menu from which the user can execute the actions handled
        by the other methods in this class.

        Args:
            None.

        Returns:
            Nothing.

        """

        seed(1)

        print('\n=================================')
        print('   Pima Indians Diabetes dataset')
        print('=================================')

        self.data_preprocessing()

        returned_from_function = True

        while True:

            if returned_from_function == True:
                print('\nChoose the action:')
                print('\n1. Classify new data.')
                print('2. Calculate algorithm accuracy.')
                print('3. Show dataset description.')
                print('4. Show dataset rows.')
                print('5. Go back to the main menu.\n')

            returned_from_function = False

            choice = input('Choice: ')

            if choice not in ['1', '2', '3', '4', '5']:
                print('Wrong choice! Please choose option 1-5.')

            elif choice == '1':

                try:
                    self.classify_data()
                    returned_from_function = True
                    continue
                except KeyboardInterrupt:
                    returned_from_function = True
                    continue

            elif choice == '2':

                try:
                    self.calculate_accuracy()
                    returned_from_function = True
                    continue
                except KeyboardInterrupt:
                    returned_from_function = True
                    continue

            elif choice == '3':

                try:
                    self.show_dataset_description()
                    returned_from_function = True
                    continue
                except KeyboardInterrupt:
                    returned_from_function = True
                    continue

            elif choice == '4':

                try:
                    self.show_dataset_rows()
                    returned_from_function = True
                    continue
                except KeyboardInterrupt:
                    returned_from_function = True
                    continue

            elif choice == '5':
                break

            else:
                raise


def main():

    pid = PimaIndiansDiabetes()
    pid.run()


if __name__ == "__main__":

    try:
        main()
    except:
        print('\nAn error has occurred during the program execution!\n')

