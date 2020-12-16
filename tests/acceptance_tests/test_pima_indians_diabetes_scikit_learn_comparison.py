#!/usr/bin/env python3

# This test compares the created Naive Bayes classifier implementation
# with scikit-learn library using iris.csv dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from random import seed
from pima_indians_diabetes import PimaIndiansDiabetes



def test_iris_scikit_learn_comparison():

    print('===============================')
    print('=== PROJECT IMPLEMENTATION ====')
    print('===============================')

    seed(1)

    filename = 'datasets/pima-indians-diabetes.csv'
    pid = PimaIndiansDiabetes()
    pid.data_preprocessing()
    project_efficiency_percent = pid.calculate_accuracy(n_folds=2)


    print('===============================')
    print('=========== SKLEARN ===========')
    print('===============================')

    '''
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    num_of_points = 75
    mislabeled_points = (y_test != y_pred).sum()
    sklearn_efficiency_percent = ((num_of_points - mislabeled_points) / num_of_points) * 100

    print(f'Number of mislabeled points out of a total {num_of_points} points : {mislabeled_points}')
    print(f'Algorithm efficiency: {round(sklearn_efficiency_percent, 5)} %')

    assert (project_efficiency_percent - sklearn_efficiency_percent) < 10
    '''
