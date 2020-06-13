import numpy as np
import os


def dataset_generator(n_datasets, rn_seed, train_size, test_size, n_features, n_predictive_features, np_low, np_high,
                      expression):
    '''
    n_datasets = Number of datasets needed to be generated
    rn_seed = Seed for numpy random
    train_size = Number of samples in training dataset
    test_size = Number of samples in testing dataset
    n_features = Total features in the dataset (Predictive + Non-predictive)
    n_predictive_features = Number of predictive elements
    np_low = Lower bound for np.random.uniform(low,high,size)
    np_high = upper bound for np.random.uniform(low,high,size)
    expression = Mathematical expression
    '''

    # Set a seed for reproducibility
    np.random.seed(rn_seed)

    # Create output folder
    outputPath = os.getcwd() + "/Regression(" + expression + ")-Datasets/"
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    # Generate dataset in iteration
    for dataset in range(1, n_datasets + 1):
        dataset_name = 'Regression(' + expression + ')-Datasets/' + str(
            dataset) + '-Regression(' + expression + ')_Train' + '.txt'
        dataset_file = open(dataset_name, 'w')

        headerList = 'N0' + '\t' + 'N1' + '\t' + 'N2' + '\t' + 'N3' + '\t' + 'N4' + '\t' + 'N5' + '\t' + 'N6' + '\t' + 'N7' + '\t' + 'N8' + '\t' + 'N9' + '\t' + 'N10' + '\t' + 'N11' + '\t' + 'N12' + '\t' + 'N13' + '\t' + 'N14' + '\t' + 'N15' + '\t' + 'N16' + '\t' + 'N17' + '\t' + 'N18' + '\t' + 'X' + '\t' + 'Class\n'
        dataset_file.write(headerList)

        # Generate training dataset
        dataset_train = np.random.uniform(low=np_low, high=np_high, size=[n_features, train_size])

        # Class for training dataset
        class_train = dataset_class(dataset_train, expression)
        class_train = np.reshape(class_train, (1, 1000))

        # Combined training dataset
        combined_train = np.transpose(np.concatenate((dataset_train, class_train), axis=0))

        dataset_file.write("\n".join("\t".join(map(str, x)) for x in combined_train))

        ## Testing dataset -----------------------------------------------------------
        test_dataset_name = 'Regression(' + expression + ')-Datasets/' + str(
            dataset) + '-Regression(' + expression + ')_Test' + '.txt'
        test_dataset_file = open(test_dataset_name, 'w')

        headerList = 'N0' + '\t' + 'N1' + '\t' + 'N2' + '\t' + 'N3' + '\t' + 'N4' + '\t' + 'N5' + '\t' + 'N6' + '\t' + 'N7' + '\t' + 'N8' + '\t' + 'N9' + '\t' + 'N10' + '\t' + 'N11' + '\t' + 'N12' + '\t' + 'N13' + '\t' + 'N14' + '\t' + 'N15' + '\t' + 'N16' + '\t' + 'N17' + '\t' + 'N18' + '\t' + 'X' + '\t' + 'Class\n'
        test_dataset_file.write(headerList)

        # Generate testing dataset
        dataset_test = np.random.uniform(low=np_low, high=np_high, size=[n_features, test_size])

        # Class for testing dataset
        class_test = dataset_class(dataset_test, expression)
        class_test = np.reshape(class_test, (1, 500))

        # Combined testing dataset
        combined_test = np.transpose(np.concatenate((dataset_test, class_test), axis=0))

        test_dataset_file.write("\n".join("\t".join(map(str, x)) for x in combined_test))


def dataset_class(dataset, expression):
    n_features = dataset.shape[0]
    predictive_elements = dataset[n_features - 1, :]
    predictive_elements2 = dataset[n_features - 2, :]

    if expression == '2X+5':
        dataset_class = 2 * predictive_elements + 5

    elif expression == '2X':
        dataset_class = 2 * predictive_elements

    elif expression == 'SinX+CosX':
        dataset_class = np.sin(predictive_elements) + np.cos(predictive_elements)

    elif expression == 'X^4+X^3+X^2+X+1':
        dataset_class = (predictive_elements ** 4) + (predictive_elements ** 3) + (
                predictive_elements ** 2) + predictive_elements + 1

    elif expression == 'CosX':
        dataset_class = np.cos(predictive_elements)

    elif expression == "X1*X2":
        dataset_class = predictive_elements * predictive_elements2

    return dataset_class


# def main():
dataset_generator(n_datasets=30, rn_seed=2, train_size=1000, test_size=500, n_predictive_features=2,
                  n_features=20, np_low=-100.0, np_high=100.0, expression='X1*X2')
