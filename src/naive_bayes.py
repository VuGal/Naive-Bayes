#!/usr/bin/env python3

from csv import reader
from math import pi, sqrt, exp
from random import randrange


class NaiveBayesClassifier:

    """

    Implementation of a Naive Bayes Classifier algorithm.

    The implementation consists of the following steps:\n
    1. Dataset values separation by class\n
    2. Calculate the arithmetic mean, standard deviation and values count for each dataset column\n
    3. Separate the calculated parameters by class\n
    4. Calculate the Gaussian Probability Density Function\n
    5. Calculate the probabilities of data belonging to each data class

    """
    
    def load_dataset_from_csv(self, csv_file):

        """

        Loads dataset from .csv file.

        Args:
            csv_file (string)
                String representing .csv filename.

        Returns:
            dataset (list[string])

        """

        dataset = list()

        with open(csv_file, 'r') as f:

            csv_reader = reader(f)

            for row in csv_reader:
                if row:
                    dataset.append(row)

            return dataset


    def map_class_names_to_ints(self, dataset, column, numbers_already=False):

        """

        Data preprocessing - maps class names (strings) to integers.

        Args:
            dataset (list)
                List representing the dataset.
            column (int)
                Number of column to be converted.
            numbers_already (bool)
                If the class names strings are numbers already, maps them to integers representing the same numbers.
                Default: False

        Returns:
            data (dict[string, int])
                Dictionary mapping class names (strings) to integers.

        """

        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        data = dict()

        if not numbers_already:
            for i, value in enumerate(unique):
                data[value] = i

        else:
            for value in unique:
                data[value] = int(value)

        for row in dataset:
            row[column] = data[row[column]]

        return data


    def convert_class_values_to_floats(self, dataset, column):

        """

        Data preprocessing - converts class values (strings) to floats to enable more efficient processing
        of the data by the program.

        Args:
            dataset (list)
                List representing the dataset.
            column (int)
                Number of column to be converted.

        Returns:
            Nothing.

        """

        for row in dataset:
            row[column] = float(row[column].strip())


    def divide_data_by_class(self, dataset):

        """

        Divides the dataset into rows belonging to the specific classes.

        Args:
            dataset (list)
                List representing the dataset.

        Returns:
            divided_data (dict[int, list])

        """

        divided_data = dict()

        for i in range(len(dataset)):

            vector = dataset[i]
            class_name = vector[-1]

            if class_name not in divided_data:
                divided_data[class_name] = list()

            divided_data[class_name].append(vector)

        return divided_data


    def arithmetic_mean(self, numbers):

        """

        Calculates the arithmetic mean.

        Args:
            numbers (list[float])
                List of float numbers.

        Returns:
            (float)
                Calculated arithmetic mean of inputted numbers.


        """

        return (sum(numbers) / float(len(numbers)))


    def std_deviation(self, numbers):

        """

        Calculates the standard deviation.

        Args:
            numbers (list[float])
                List of float numbers.

        Returns:
            (float)
                Calculated standard deviation of inputted numbers.


        """

        mean = self.arithmetic_mean(numbers)
        variance = sum([(x - mean) ** 2 for x in numbers]) / float(len(numbers) - 1)

        return sqrt(variance)


    def gather_data_params(self, dataset):

        """

        Calculates the parameters (arithmetic mean, standard deviation, values count) for each dataset column.

        Args:
            dataset (list)
                List representing the dataset.

        Returns:
            data_params (list[float, float, int])
                List of lists containing the parameters.

        """

        data_params = [(self.arithmetic_mean(column), self.std_deviation(column), len(column)) for column in
                       zip(*dataset)]
        del(data_params[-1])

        return data_params


    def calculate_class_parameters(self, dataset):

        """

        Divides the dataset by class, then calculate the parameters for each class.

        Args:
            dataset (list)
                List representing the dataset.

        Returns:
            class_params (dict[int, [float, float, int]])
                Dictionary mapping class names to parameters lists.

        """

        divided_data = self.divide_data_by_class(dataset)
        class_params = dict()

        for class_name, rows in divided_data.items():
            class_params[class_name] = self.gather_data_params(rows)

        return class_params


    def gaussian_probability(self, x, mean, std_dev):

        """

        Calculates the Gaussian probability distribution function.

        Args:
            x (list)
                List representing the dataset.
            mean (float)
                Arithmetic mean.
            std_dev (float)
                Standard deviation.

        Returns:
            params_by_class (dict[int, [float, float, int]])
                List of lists containing the parameters.

        """

        exponent = exp(-((x - mean) ** 2 / (2 * std_dev ** 2)))

        return (1 / (sqrt(2 * pi) * std_dev)) * exponent


    def calculate_class_probabilities(self, divided_data, row):

        """

        Calculates the probabilities of classifying a specified data row to each class.

        Args:
            divided_data (list)
                List representing the dataset.
            row (int)
                Row of data which will be classified.

        Returns:
            probabilities (dict[int, float])
                List of lists containing the parameters.

        """

        total_rows = sum([divided_data[label][0][2] for label in divided_data])

        probabilities = dict()

        for class_name, class_params in divided_data.items():

            probabilities[class_name] = divided_data[class_name][0][2] / float(total_rows)

            for i in range(len(class_params)):
                mean, std_dev, _ = class_params[i]
                probabilities[class_name] *= self.gaussian_probability(row[i], mean, std_dev)

        return probabilities


    def predict(self, divided_data, row):

        """

        Predicts a class for a given data row.

        Args:
            divided_data (list)
                List representing the dataset.
            row (int)
                Row of data which will be classified.

        Returns:
            predicted_label (float)
                Float representing the predicted class name.

        """

        probabilities = self.calculate_class_probabilities(divided_data, row)
        predicted_label, predicted_probability = None, -1

        for class_value, probability in probabilities.items():

            if predicted_label is None or probability > predicted_probability:
                predicted_probability = probability
                predicted_label = class_value

        return predicted_label


    def k_fold_cross_validation_split(self, dataset, folds_num):

        """

        Data resampling - k-fold cross validation split.

        Args:
            dataset (list)
                List representing the dataset.
            folds_num (int)
                Number of folds used in the algorithm.

        Returns:
            dataset_split (float)
                List representing the dataset after the k-split.

        """

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


    def measure_algorithm_accuracy(self, actual, predicted):

        """

        Measures algorithm accuracy in percent.

        Args:
            actual (int)
                Integer representing the actual class name.
            predicted (int)
                Integer representing the predicted class name.

        Returns:
            (float)
                The algorithm accuracy in percent.

        """

        correct = 0

        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1

        return correct / float(len(actual)) * 100.0


    def evaluate_algorithm(self, dataset, n_folds):

        """

        Evaluates the algorithm using a k-fold cross validation split.

        Args:
            dataset (list)
                List representing the dataset.
            n_folds (int)
                Number of folds used in the algorithm.

        Returns:
            scores (list[float])
                List containing the classification accuracies in each fold.

        """

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

            params_by_class = self.calculate_class_parameters(train_set)
            predicted = list()

            for row in test_set:
                output = self.predict(params_by_class, row)
                predicted.append(output)

            actual = [row[-1] for row in fold]
            accuracy = self.measure_algorithm_accuracy(actual, predicted)
            scores.append(accuracy)

        return scores
