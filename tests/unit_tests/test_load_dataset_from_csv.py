from naive_bayes import NaiveBayesClassifier


def test_load_dataset_from_csv():

    classifier = NaiveBayesClassifier()

    csv_filename = 'iris.csv'
    data_0 = ['5.1', '3.5', '1.4', '0.2', 'Iris-setosa']
    data_2 = ['4.7', '3.2', '1.3', '0.2', 'Iris-setosa']
    data_40 = ['5.1','3.4','1.5','0.2','Iris-setosa']
    data_61 = ['5.0','2.0','3.5','1.0','Iris-versicolor']
    data_80 = ['5.5','2.4','3.7','1.0','Iris-versicolor']
    data_90 = ['5.5','2.5','4.0','1.3','Iris-versicolor']
    data_105 = ['6.5','3.0','5.8','2.2','Iris-virginica']
    data_111 = ['6.5','3.2','5.1','2.0','Iris-virginica']
    data_126 = ['7.2', '3.2', '6.0', '1.8', 'Iris-virginica']
    data_144 = ['6.8','3.2','5.9','2.3','Iris-virginica']

    readed_dataset = classifier.load_dataset_from_csv(csv_filename)

    assert readed_dataset[0] == data_0
    assert readed_dataset[2] == data_2
    assert readed_dataset[40] == data_40
    assert readed_dataset[61] == data_61
    assert readed_dataset[80] == data_80
    assert readed_dataset[90] == data_90
    assert readed_dataset[105] == data_105
    assert readed_dataset[111] == data_111
    assert readed_dataset[126] == data_126
    assert readed_dataset[144] == data_144

