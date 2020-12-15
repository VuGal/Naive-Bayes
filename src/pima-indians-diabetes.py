#!/usr/bin/env python3

# This file holds the Naive Bayes classifier implementation for 'pima-indians-diabetes.csv' dataset.

from random import seed
from naive_bayes import NaiveBayesClassifier


def main():

    seed(1)

    nbc = NaiveBayesClassifier()

    filename = '../datasets/pima-indians-diabetes.csv'
    dataset = nbc.load_dataset_from_csv(filename)

    for i in range(len(dataset[0]) - 1):
        nbc.string_column_to_float(dataset, i)

    nbc.string_column_to_int(dataset, len(dataset[0]) - 1)

    n_folds = 5
    scores = nbc.evaluate_algorithm(dataset, n_folds)

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


if __name__ == "__main__":

    try:
        main()
    except:
        print('\nAn error has occurred during the program execution!\n')

