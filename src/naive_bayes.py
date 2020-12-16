#!/usr/bin/env python3

from csv import reader
from math import pi, sqrt, exp
from random import randrange


class NaiveBayesClassifier:


    # Load dataset from .csv file
    def load_dataset_from_csv(self, csv_file):

        dataset = list()

        with open(csv_file, 'r') as f:

            csv_reader = reader(f)

            for row in csv_reader:
                if row:
                    dataset.append(row)

            return dataset


    # Data preprocessing - change strings to ints (useful for changing class names to numbers)
    def string_column_to_int(self, dataset, column):

        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        data = dict()

        for i, value in enumerate(unique):
            data[value] = i
            print(f'{value} => {i}')

        for row in dataset:
            row[column] = data[row[column]]

        return data


    # Data preprocessing - change strings to floats (useful for changing string class values to floats,
    # which are easier to process by the program later)
    def string_column_to_float(self, dataset, column):

        for row in dataset:
            row[column] = float(row[column].strip())


    # Divide the dataset into values belonging to the specific classes
    def divide_data_by_class(self, dataset):

        divided_data = dict()

        for i in range(len(dataset)):

            vector = dataset[i]
            class_name = vector[-1]

            if class_name not in divided_data:
                divided_data[class_name] = list()

            divided_data[class_name].append(vector)

        return divided_data


    # Calculate the arithmetic mean
    def arithmetic_mean(self, numbers):

        return (sum(numbers) / float(len(numbers)))


    # Calculate the standard deviation
    def std_deviation(self, numbers):

        mean = self.arithmetic_mean(numbers)
        variance = sum([(x - mean) ** 2 for x in numbers]) / float(len(numbers) - 1)

        return sqrt(variance)


    # Calculate the parameters (arithmetic mean, standard deviation, values count) for each dataset column,
    # return a list of lists containing the parameters
    def gather_data_params(self, dataset):

        data_params = [(self.arithmetic_mean(column), self.std_deviation(column), len(column)) for column in
                       zip(*dataset)]
        del(data_params[-1])

        return data_params


    # Divide the dataset by class, then calculate the parameters for each row
    def divide_data_params_by_class(self, dataset):

        divided_data = self.divide_data_by_class(dataset)
        params_by_class = dict()

        for class_name, rows in divided_data.items():
            params_by_class[class_name] = self.gather_data_params(rows)

        return params_by_class


    # Calculate the Gaussian probability distribution function
    def gaussian_probability(self, x, mean, std_dev):

        exponent = exp(-((x - mean) ** 2 / (2 * std_dev ** 2)))

        return (1 / (sqrt(2 * pi) * std_dev)) * exponent


    # Calculate the probabilities of classifying a specified data row to each class
    def calculate_class_probabilities(self, divided_data, row):

        total_rows = sum([divided_data[label][0][2] for label in divided_data])

        probabilities = dict()

        for class_name, class_params in divided_data.items():

            probabilities[class_name] = divided_data[class_name][0][2] / float(total_rows)

            for i in range(len(class_params)):
                mean, std_dev, _ = class_params[i]
                probabilities[class_name] *= self.gaussian_probability(row[i], mean, std_dev)

        return probabilities


    # Predict a class for a given data row
    def predict(self, divided_data, row):

        probabilities = self.calculate_class_probabilities(divided_data, row)
        predicted_label, predicted_probability = None, -1

        for class_value, probability in probabilities.items():

            if predicted_label is None or probability > predicted_probability:
                predicted_probability = probability
                predicted_label = class_value

        return predicted_label


    # Data resampling - k-fold cross validation split
    def k_fold_cross_validation_split(self, dataset, folds_num):

        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / folds_num)

        for _ in range(folds_num):

            fold = list()

            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))

            dataset_split.append(fold)

        return dataset_split


    # Measure algorithm accuracy in percent
    def measure_algorithm_accuracy(self, actual, predicted):

        correct = 0

        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1

        return correct / float(len(actual)) * 100.0


    # Evaluate the algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, n_folds):

        folds = self.k_fold_cross_validation_split(dataset, n_folds)
        scores = list()

        for fold in folds:

            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])

            test_set = list()

            for row in fold:
                row_copy = list(row)
                row_copy[-1] = None
                test_set.append(row_copy)

            params_by_class = self.divide_data_params_by_class(train_set)
            predicted = list()

            for row in test_set:
                output = self.predict(params_by_class, row)
                predicted.append(output)

            actual = [row[-1] for row in fold]
            accuracy = self.measure_algorithm_accuracy(actual, predicted)
            scores.append(accuracy)

        return scores

