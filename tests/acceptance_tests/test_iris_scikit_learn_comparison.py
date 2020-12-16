#!/usr/bin/env python3

# This test compares the created Naive Bayes classifier implementation
# with scikit-learn library using iris.csv dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from naive_bayes import NaiveBayesClassifier


def test_iris_scikit_learn_comparison():

    print('===============================')
    print('=== PROJECT IMPLEMENTATION ====')
    print('===============================')

    #TODO: Recreate the same conditions on project implementation (test_size == 0.5)





    print('===============================')
    print('=========== SKLEARN ===========')
    print('===============================')

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    num_of_points = 75
    mislabeled_points = (y_test != y_pred).sum()
    efficiency_percent = ((num_of_points - mislabeled_points) / num_of_points) * 100

    assert efficiency_percent > 90

    print(f'Number of mislabeled points out of a total {num_of_points} points : {mislabeled_points}')
    print(f'Algorithm efficiency: {round(efficiency_percent, 5)} %')

    #TODO: Change to pytest format

