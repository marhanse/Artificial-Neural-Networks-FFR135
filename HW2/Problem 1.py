# # # # # # # # # # # # # # # # # # # # # # # # #
# Homework 2                                    #
# Problem 1:  Perceptron with one hidden layer  #
# Author: Marcus Hansen                         #
# Course: FFR135 Artificial Neural Networks     #
# # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import random
import sys
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)

# Open csv file and convert to numpy
df_training_set = pd.read_csv('training_set.csv')
df_validation_set = pd.read_csv('validation_set.csv')
training_set = df_training_set.to_numpy()
validation_set = df_validation_set.to_numpy()


# Function for centering each of the two input components and normalizing their respective variances to 1
def data_processing(data_set):
    x1 = data_set[:, 0]
    x2 = data_set[:, 1]
    mean_x1 = np.mean(x1)
    mean_x2 = np.mean(x2)
    std_x1 = np.std(x1)
    std_x2 = np.std(x2)
    data_set[:, 0] = (x1 - mean_x1) / std_x1
    data_set[:, 1] = (x2 - mean_x2) / std_x2

    return data_set


# Function to compute the classification error
def classification_error(output, target):
    C = np.sum(np.abs(np.sign(output) - target)) / (2 * len(output))
    return C


# Function to compute the derivative of the activation function
def g_Prime_value(local_Field):
    g_prime = (1 - np.tanh(local_Field) ** 2)
    return g_prime


def main_program(learning_rate, M1):
    processed_training = data_processing(training_set)
    processed_validation = data_processing(validation_set)

    target_training = processed_training[:, 2]
    target_validation = processed_validation[:, 2]

    pattern_training = processed_training[:, :2]
    pattern_validation = processed_validation[:, :2]

    number_Train_Patterns = np.size(processed_training, 0)
    number_Valid_Patterns = np.size(processed_validation, 0)

    train_index = list(range(number_Train_Patterns))
    valid_index = list(range(number_Valid_Patterns))

    # Initialising weights and thresholds
    weight_Matrix_1 = np.random.normal(loc=0.0, scale=1 / np.sqrt(2), size=(M1, 2))
    weight_Matrix_2 = np.random.normal(loc=0.0, scale=1 / np.sqrt(M1), size=(M1, 1))
    threshold_1 = np.zeros([M1, 1])
    threshold_2 = 0

    # Training and validation of the network
    epoch = 0
    while True:
        epoch += 1
        random.shuffle(train_index)
        for j in train_index:
            p_training = pattern_training[j, :].reshape(2, 1)

            # Forward
            local_field_1 = -threshold_1 + np.matmul(weight_Matrix_1, p_training)
            layer_output = np.tanh(local_field_1)
            local_field_2 = -threshold_2 + np.matmul(weight_Matrix_2.T, layer_output)
            Output = np.tanh(local_field_2)

            # Backward
            target = target_training[j]
            error_2 = (target - Output) * g_Prime_value(local_field_2)
            delta_weight_Matrix_2 = learning_rate * error_2 * layer_output
            error_1 = np.matmul(weight_Matrix_2, error_2) * g_Prime_value(local_field_1)
            delta_weight_Matrix_1 = np.matmul(error_1, p_training.T)

            # Update
            weight_Matrix_1 += delta_weight_Matrix_1
            weight_Matrix_2 += delta_weight_Matrix_2

            threshold_1 -= learning_rate * error_1
            threshold_2 -= learning_rate * error_2

        output_epoch = np.zeros([len(target_validation), 1])
        target_epoch = np.zeros([len(target_validation), 1])

        # Validation
        for i in range(len(valid_index)):
            p_validation = pattern_validation[valid_index[i], :].reshape(2, 1)
            local_field_1 = - threshold_1 + np.matmul(weight_Matrix_1, p_validation)
            layer_output = np.tanh(local_field_1)
            local_field_2 = - threshold_2 + np.matmul(weight_Matrix_2.T, layer_output)
            Output = np.tanh(local_field_2)
            output_epoch[i, :] = Output
            target_epoch[i, :] = target_validation[valid_index[i]]

        c = (classification_error(output_epoch, target_epoch))
        print(f"Epoch: {epoch}, Classification error: {c}")

        if c < 0.12:
            # Convert to dataframe and then save as csv file
            df_1 = pd.DataFrame(weight_Matrix_1)
            df_2 = pd.DataFrame(weight_Matrix_2)
            df_3 = pd.DataFrame(threshold_1)
            df_4 = pd.DataFrame(threshold_2)
            df_1.to_csv("w1.csv", index=False, header=False)
            df_2.to_csv("w2.csv", index=False, header=False)
            df_3.to_csv("t1.csv", index=False, header=False)
            df_4.to_csv("t2.csv", index=False, header=False)
            print('')
            print('Classification error is now below 12 %')
            break


# Hyperparameters
neurons_hidden = 50
rate = 0.01

main_program(rate, neurons_hidden)
