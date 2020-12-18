from naive_bayes import NaiveBayesClassifier


def test_arithmetic_mean():

    classifier = NaiveBayesClassifier()
    assert classifier.arithmetic_mean(numbers=[1, 2, 3, 4, 5, 6, 7]) == 4
