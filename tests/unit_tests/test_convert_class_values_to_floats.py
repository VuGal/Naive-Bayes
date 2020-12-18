from naive_bayes import NaiveBayesClassifier
import random


def test_convert_class_values_to_floats():

    classifier = NaiveBayesClassifier()

    dataset = [['3.393533211', '2.331273381', '0'],
               ['3.110073483', '1.781539638', '0'],
               ['1.343808831', '3.368360954', '0'],
               ['3.582294042', '4.67917911', '0'],
               ['2.280362439', '2.866990263', '0'],
               ['7.423436942', '4.696522875', '1'],
               ['5.745051997', '3.533989803', '1'],
               ['9.172168622', '2.511101045', '1'],
               ['7.792783481', '3.424088941', '1'],
               ['7.939820817', '0.791637231', '1']]

    for i in range(0, len(dataset[0])-1):
        classifier.convert_class_values_to_floats(dataset, i)

    for i in range(0, len(dataset)):
        tested_row = random.randint(0, len(dataset)-1)
        tested_column = random.randint(0, len(dataset[0]) - 2)
        assert isinstance(dataset[tested_row][tested_column], float)
