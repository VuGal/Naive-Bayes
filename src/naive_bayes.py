#!/usr/bin/env python3

from math import pi, sqrt, exp
from csv import reader


def divide_data_by_class(dataset):

    divided_data = dict()

    for i in range(len(dataset)):

        vector = dataset[i]
        class_name = vector[-1]

        if class_name not in divided_data:
            divided_data[class_name] = list()

        divided_data[class_name].append(vector)

    return divided_data


class NaiveBayesClassifier:


    def load_dataset_from_csv(self, csv_file):

        dataset = list()

        with open(csv_file, 'r') as f:

            csv_reader = reader(f)

            for row in csv_reader:
                if row:
                    dataset.append(row)

            return dataset


    def string_column_to_float(self, dataset, column):

        for row in dataset:
            row[column] = float(row[column].strip())


    def string_column_to_int(self, dataset, column):

        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        data = dict()

        for i, value in enumerate(unique):
            data[value] = i
            print(f'[{value}] => {i}')

        for row in dataset:
            row[column] = data[row[column]]

        return data


    def arithmetic_mean(self, numbers):

        return (sum(numbers) / float(len(numbers)))


    def std_deviation(self, numbers):

        mean = self.arithmetic_mean(numbers)
        variance = sum([(x - mean) ** 2 for x in numbers]) / float(len(numbers) - 1)

        return sqrt(variance)


    def gather_data_params(self, dataset):

        data_params = [(self.arithmetic_mean(column), self.std_deviation(column), len(column)) for column in
                       zip(*dataset)]
        del (data_params[-1])

        return data_params


    def divide_data_params_by_class(self, dataset):

        divided_data = divide_data_by_class(dataset)
        params_by_class = dict()

        for class_value, rows in divided_data.items():
            params_by_class[class_value] = self.gather_data_params(rows)

        return params_by_class


    def gaussian_probability(self, x, mean, std_dev):

        exponent = exp(-((x - mean) ** 2 / (2 * std_dev ** 2)))

        return (1 / (sqrt(2 * pi) * std_dev)) * exponent


    def calculate_class_probabilities(self, divided_data, row):

        total_rows = sum([divided_data[label][0][2] for label in divided_data])

        probabilities = dict()

        for class_value, class_params in divided_data.items():

            probabilities[class_value] = divided_data[class_value][0][2] / float(total_rows)

            for i in range(len(class_params)):
                mean, std_dev, _ = class_params[i]
                probabilities[class_value] *= self.gaussian_probability(row[i], mean, std_dev)

            return probabilities


    def predict(self, divided_data, row):

        probabilities = self.calculate_class_probabilities(divided_data, row)
        predicted_label, predicted_probability = None, -1

        for class_value, probability in probabilities.items():

            if predicted_label is None or probability > predicted_probability:
                predicted_probability = probability
                predicted_label = class_value

        return predicted_label


def main():
    nbc = NaiveBayesClassifier()

    # Make a prediction with Naive Bayes on Iris Dataset
    filename = 'iris.csv'
    dataset = nbc.load_dataset_from_csv(filename)

    for i in range(len(dataset[0]) - 1):
        nbc.string_column_to_float(dataset, i)

    # convert class column to integers
    nbc.string_column_to_int(dataset, len(dataset[0]) - 1)
    # fit model
    model = nbc.divide_data_params_by_class(dataset)
    # define a new record
    row = [5.7, 2.9, 4.2, 1.3]
    # predict the label
    label = nbc.predict(model, row)
    print('Data=%s, Predicted: %s' % (row, label))


if __name__ == "__main__":

    try:
        main()
    except:
        print('\nPodczas dzialania programu wystapil blad!\n')
