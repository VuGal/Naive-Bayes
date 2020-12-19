#!/usr/bin/env python3

from csv import reader
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from random import seed
from pima_indians_diabetes import PimaIndiansDiabetes


def test_pima_indians_diabetes_scikit_learn_comparison():

    """

    This test compares the created Naive Bayes classifier implementation
    with scikit-learn library using pima-indians-diabetes.csv dataset.

    """

    print('\n===============================')
    print('=== PROJECT IMPLEMENTATION ====')
    print('===============================')

    seed(1)

    pid = PimaIndiansDiabetes()
    pid.data_preprocessing()
    project_efficiency_percent = pid.calculate_accuracy(n_folds=2)


    print('\n===============================')
    print('=========== SKLEARN ===========')
    print('===============================')

    # loading data from .csv file
    filename = 'datasets/pima-indians-diabetes.csv'
    X = list()
    y = list()

    with open(filename, 'r') as f:

        csv_reader = reader(f)

        for i, row in enumerate(csv_reader):
            converted_row = list()
            for j in range(len(row)-1):
                converted_row.append(float(row[j]))
            X.append(converted_row)
            y.append(int(row[-1]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    num_of_points = 384
    mislabeled_points = (y_test != y_pred).sum()
    sklearn_efficiency_percent = ((num_of_points - mislabeled_points) / num_of_points) * 100

    print(f'\n\nCalculating the scikit-learn algorithm accuracy with pima-indians-diabetes.csv dataset...')
    print(f'\nNumber of mislabeled points out of a total {num_of_points} points : {mislabeled_points}')
    print(f'\nAlgorithm efficiency: {round(sklearn_efficiency_percent, 5)} %')

    assert (project_efficiency_percent - sklearn_efficiency_percent) < 10


def main():

    test_pima_indians_diabetes_scikit_learn_comparison()


if __name__ == "__main__":

    try:
        main()
    except:
        print('\nAn error has occurred during the program execution!\n')
