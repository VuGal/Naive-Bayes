#!/usr/bin/env python3

# This file holds the Naive Bayes classifier implementation for 'iris.csv' dataset.

from random import seed
from naive_bayes import NaiveBayesClassifier


class Iris:


    def run(self):

        seed(1)
        print()

        nbc = NaiveBayesClassifier()

        filename = 'datasets/iris.csv'
        dataset = nbc.load_dataset_from_csv(filename)

        for i in range(len(dataset[0]) - 1):
            nbc.string_column_to_float(dataset, i)

        nbc.string_column_to_int(dataset, len(dataset[0]) - 1)

        print('\nTrying to classify the new data: [5.6, 3.1, 4.0, 1.2]')

        model = nbc.divide_data_params_by_class(dataset)
        row = [5.6, 3.1, 4.0, 1.2]
        label = nbc.predict(model, row)

        print(f'\nPredicted: {label}')

        n_folds = 5
        scores = nbc.evaluate_algorithm(dataset, n_folds)

        print('\n\nCalculating the accuracy of the classifier using the iris.csv dataset...')
        print('\nResampling: k-fold cross validation split')
        print('\nAccuracy (5 folds): %.3f%%\n' % (sum(scores) / float(len(scores))))


def main():

    iris = Iris()
    iris.run()


if __name__ == "__main__":

    try:
        main()
    except:
        print('\nAn error has occurred during the program execution!\n')

