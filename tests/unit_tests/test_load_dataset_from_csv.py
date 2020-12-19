from naive_bayes import NaiveBayesClassifier


def test_load_dataset_from_csv():

    classifier = NaiveBayesClassifier()

    csv_filename = 'datasets/iris.csv'

    data_0 = ['5.1', '3.5', '1.4', '0.2', 'Iris-setosa']
    data_2 = ['4.7', '3.2', '1.3', '0.2', 'Iris-setosa']
    data_39 = ['5.1','3.4','1.5','0.2','Iris-setosa']
    data_60 = ['5.0','2.0','3.5','1.0','Iris-versicolor']
    data_81 = ['5.5','2.4','3.7','1.0','Iris-versicolor']
    data_89 = ['5.5','2.5','4.0','1.3','Iris-versicolor']
    data_104 = ['6.5','3.0','5.8','2.2','Iris-virginica']
    data_110 = ['6.5','3.2','5.1','2.0','Iris-virginica']
    data_125 = ['7.2', '3.2', '6.0', '1.8', 'Iris-virginica']
    data_143 = ['6.8','3.2','5.9','2.3','Iris-virginica']

    readed_dataset = classifier.load_dataset_from_csv(csv_filename)

    assert readed_dataset[0] == data_0
    assert readed_dataset[2] == data_2
    assert readed_dataset[39] == data_39
    assert readed_dataset[60] == data_60
    assert readed_dataset[81] == data_81
    assert readed_dataset[89] == data_89
    assert readed_dataset[104] == data_104
    assert readed_dataset[110] == data_110
    assert readed_dataset[125] == data_125
    assert readed_dataset[143] == data_143

    csv_filename_2 = 'tests/unit_tests/resources/load_test.csv'

    readed_dataset_2 = classifier.load_dataset_from_csv(csv_filename_2)

    assert len(readed_dataset_2) == 3
