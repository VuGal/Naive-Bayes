from naive_bayes import NaiveBayesClassifier


def test_measure_algorithm_accuracy():

    classifier = NaiveBayesClassifier

    actual = [1, 2, 4, 5, 6]
    predicted = [1, 2, 4, 5, 6]

    assert classifier.measure_algorithm_accuracy(classifier, actual, predicted) == 100
    assert isinstance(classifier.measure_algorithm_accuracy(classifier, actual, predicted), float)
