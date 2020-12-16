from naive_bayes import NaiveBayesClassifier


def test_predict():

    classifier = NaiveBayesClassifier()

    dataset = {1: [(2.7420144012, 0.9265683289298018, 5), (3.0054686692, 1.1073295894898725, 5)],
               0: [(7.6146523718, 1.2344321550313704, 5), (2.9914679790000003, 1.4541931384601618, 5)]}

    row = [3.7, 2.9, 0]

    results_predict = classifier.predict(dataset, row)

    assert results_predict == 1
