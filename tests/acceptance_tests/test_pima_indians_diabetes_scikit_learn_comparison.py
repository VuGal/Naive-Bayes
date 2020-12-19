#!/usr/bin/env python3

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

    filename = 'datasets/pima-indians-diabetes.csv'
    pid = PimaIndiansDiabetes()
    pid.data_preprocessing()
    project_efficiency_percent = pid.calculate_accuracy(n_folds=2)


    print('\n===============================')
    print('=========== SKLEARN ===========')
    print('===============================')

    #TODO: Implement the scikit-learn version of the pima-indians-diabetes.csv accuracy calculation

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
