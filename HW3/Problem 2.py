# # # # # # # # # # # # # # # # # # # # # # # # #
# Homework 3                                    #
# Problem 2:  Chaotic time-series prediction    #
# Author: Marcus Hansen                         #
# Course: FFR135 Artificial Neural Networks     #
# # # # # # # # # # # # # # # # # # # # # # # # #


import numpy as np
import sys
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)

df_training_set = pd.read_csv('training-set.csv', header=None, index_col=None)
df_test_set = pd.read_csv('test-set-5.csv', header=None, index_col=None)
training_set = df_training_set.to_numpy()
test_set = df_test_set.to_numpy()

inputNeurons = 3
reservoirNeurons = 500
stepsToPredict = 500
timesTraining = np.size(training_set, 1)
timesTest = np.size(test_set, 1)
k = 0.01

weightMatrixIN = np.random.normal(loc=0.0, scale=np.sqrt(0.002), size=(inputNeurons, reservoirNeurons))
weightMatrixReservoir = np.random.normal(loc=0.0, scale=np.sqrt(2 / reservoirNeurons), size=(reservoirNeurons, reservoirNeurons))
rInitial = np.random.uniform(0, 1, size=(1, reservoirNeurons))
X = np.zeros([timesTraining-1, reservoirNeurons])

targets = training_set[:, 1:]

for t in range(timesTraining-1):
    inputData = training_set[:, t].T.reshape(inputNeurons, 1)
    argument = np.matmul(weightMatrixReservoir, rInitial.T) + np.matmul(weightMatrixIN.T, inputData)
    r = np.tanh(argument).T
    X[t, :] = r

ridgeParameterMatrix = np.identity(reservoirNeurons)*k
argument1 = np.matmul(targets, X)
argument2 = np.linalg.inv(np.matmul(X.T, X) + ridgeParameterMatrix)
weightMatrixOut = np.matmul(argument1, argument2)

inputData = test_set[:, 0].T.reshape(inputNeurons, 1)
iteration = 0
yComponent = np.zeros([1, stepsToPredict + timesTest])
while True:
    argument = np.matmul(weightMatrixReservoir, rInitial.T) + np.matmul(weightMatrixIN.T, inputData)
    r = np.tanh(argument)
    output = np.matmul(weightMatrixOut, r)
    inputData = output
    yComponent[0, iteration] = output[1 , 0]
    iteration += 1

    if iteration == stepsToPredict + timesTest:
        break

test = yComponent[0, 0:timesTest-1]
yComponent = yComponent[0, timesTest:]
df_1 = pd.DataFrame(yComponent)
df_1.to_csv("prediction.csv", index=False, header=False)


