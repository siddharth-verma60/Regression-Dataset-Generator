import random
import pandas as pd
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

    # Generate dataset in iteration
    for dataset in range(0, n_datasets):
        folderName = "/Users/siddharthverma/Exstracs_Project/Final_Outputs/Datasets/X1*X2/"
        # Generate training dataset
        dataset_train = np.random.uniform(low=np_low, high=np_high, size=[n_features, train_size])

        # Save training dataset in txt file
        dataset_train_name = folderName + 'Regression(' + expression + ')_Train_X_' + str(dataset) + '.txt'
        np.savetxt(dataset_train_name, dataset_train.transpose(), delimiter='\t')

        # Class for training dataset
        class_train = dataset_class(n_predictive_features, dataset_train, expression)
        class_train_name = folderName + 'Regression(' + expression + ')_Train_y_' + str(dataset) + '.txt'
        np.savetxt(class_train_name, class_train.transpose(), delimiter='\t')

        # # Save combined test file
        # columns = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15',
        #            'N16', 'N17', 'N18', 'X', 'Class']
        # combined_training_file_df = pd.DataFrame(
        #     np.concatenate((dataset_train.transpose(), class_train.transpose()), axis=1), columns=columns)
        # combined_training_file_df.to_csv('Regression(' + expression + ')_Train_' + str(dataset) + '.txt', sep='\t',
        #                                  index=False)

        # Generate testing dataset
        dataset_test = np.random.uniform(low=np_low, high=np_high, size=[n_features, test_size])

        # Save testing dataset in txt file
        dataset_test_name = folderName + 'Regression(' + expression + ')_Test_X_' + str(dataset) + '.txt'
        np.savetxt(dataset_test_name, dataset_test.transpose(), delimiter='\t')

        # Class for testing dataset
        class_test = dataset_class(n_predictive_features, dataset_test, expression)
        class_test_name = folderName + 'Regression(' + expression + ')_Test_y_' + str(dataset) + '.txt'
        np.savetxt(class_test_name, class_test.transpose(), delimiter='\t')

        # # Save combined test file
        # columns = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15',
        #            'N16', 'N17', 'N18', 'X', 'Class']
        # combined_test_file_df = pd.DataFrame(np.concatenate((dataset_test.transpose(), class_test.transpose()), axis=1),
        #                                      columns=columns)
        # combined_test_file_df.to_csv('Regression(' + expression + ')_Test_' + str(dataset) + '.txt', sep='\t',
        #                              index=False)


def dataset_class(n_predictive, dataset, expression):
    n_features = dataset.shape[0]
    # predictive_elements = dataset[n_features - n_predictive:n_features]
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

    elif expression == '2-X1:3+5*X2':
        dataset_class = 2 - predictive_elements / 3 + 5 * predictive_elements2

    elif expression == 'X1*X2':
        dataset_class = predictive_elements * predictive_elements2

    return dataset_class


# def main():
dataset_generator(n_datasets=15, rn_seed=2, train_size=1000, test_size=500, n_predictive_features=1,
                  n_features=20, np_low=-100.0, np_high=100.0, expression='X1*X2')
