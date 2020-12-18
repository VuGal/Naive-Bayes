from naive_bayes import NaiveBayesClassifier


def test_gaussian_probability():

    classifier = NaiveBayesClassifier()

    numbers = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    results = [0.3989422804014327, 0.24197072451914337, 0.24197072451914337]

    for i in range(0, len(numbers)):
        assert classifier.gaussian_probability(numbers[i][0], numbers[i][1], numbers[i][2]) == results[i]
