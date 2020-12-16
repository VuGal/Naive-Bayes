from naive_bayes import NaiveBayesClassifier
import pytest_mock as mocker


def test_divide_data_params_by_class():
    classifier = NaiveBayesClassifier()
    dataset = [[3.393533211, 2.331273381, 0],
               [3.110073483, 1.781539638, 0],
               [1.343808831, 3.368360954, 0],
               [3.582294042, 4.67917911, 0],
               [2.280362439, 2.866990263, 0],
               [7.423436942, 4.696522875, 1],
               [5.745051997, 3.533989803, 1],
               [9.172168622, 2.511101045, 1],
               [7.792783481, 3.424088941, 1],
               [7.939820817, 0.791637231, 1]]

    results_dataset = {0: [(2.7420144012, 0.9265683289298018, 5),
                           (3.0054686692, 1.1073295894898725, 5)],
                       1: [(7.6146523718, 1.2344321550313704, 5),
                           (2.9914679790000003, 1.4541931384601618, 5)]}

    assert classifier.divide_data_params_by_class(dataset) == results_dataset
