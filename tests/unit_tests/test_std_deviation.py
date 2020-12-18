from naive_bayes import NaiveBayesClassifier
import numpy as np


def test_std_deviation():

    classifier = NaiveBayesClassifier()
    numbers = [0.5, 1, 4.56, 3]

    assert np.around(classifier.std_deviation(numbers), 13) == 1.8728498783049
    assert classifier.std_deviation(numbers) == np.std(numbers, ddof=1)
