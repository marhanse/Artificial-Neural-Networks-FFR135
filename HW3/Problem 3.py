# # # # # # # # # # # # # # # # # # # # # # # # #
# Homework 3                                    #
# Problem 3:  Self organising map 2023          #
# Author: Marcus Hansen                         #
# Course: FFR135 Artificial Neural Networks     #
# # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import time

np.set_printoptions(threshold=sys.maxsize)


# Functions
def CalculateLearningRate(initialLearningRate, decayRate, epoch):
    learningRate = initialLearningRate * np.exp(-decayRate * epoch)
    return learningRate


def CalculateFunctionWidth(initialWidth, decayRate, epoch):
    width = initialWidth * np.exp(-decayRate * epoch)
    return width


def CalculateWinningNeuron(input, weightMatrix):
    weightMatrixRows = np.size(weightMatrix, 0)
    weightMatrixCols = np.size(weightMatrix, 1)
    minDistance = np.inf
    for row in range(weightMatrixRows):
        for col in range(weightMatrixCols):
            p1 = weightMatrix[row, col, :]
            p2 = input
            distance = np.linalg.norm(p1 - p2)
            if distance < minDistance:
                minDistance = distance
                neuron = [row, col]
    return neuron


def testing(inputDataNormalized, weightArray, class_colors, fig, axs, sup):
    # Create a scatter plot
    for i in range(numberOfInputes):
        input = inputDataNormalized[i, :]
        winner = CalculateWinningNeuron(input, weightArray)
        winnerRow = winner[0]
        winnerCol = winner[1]

        label = labels[i][0]

        color = class_colors.get(label)

        axs.scatter(winnerRow, winnerCol, c=color)
        axs.set_title(sup)
        fig.suptitle('Self Organising map before and after training')
        axs.set(xlabel='x-coordinate', ylabel='y-coordinate')


start_time = time.time()

# Data handling
dfInputData = pd.read_csv('iris-data.csv', header=None, index_col=None)
dfLabels = pd.read_csv('iris-labels.csv', header=None, index_col=None)
inputData = dfInputData.to_numpy()
labels = dfLabels.to_numpy()
numberOfInputes = np.size(inputData, 0)
maxValueDataSet = np.max(inputData)
inputDataNormalized = inputData / maxValueDataSet

# Hyperparameters and weights
weightArray = np.random.uniform(0, 1, size=(40, 40, 4))
outputArray = np.zeros([40, 40])

# Initial Parameters
initialLearningRate = 0.1
learningDecayRate = 0.01
initialFunctionWidth = 10
widthDecay = 0.05
epoch = 10

# Plot settings
class_colors = {0: 'g', 1: 'r', 2: 'b'}
fig, axs = plt.subplots(1, 2)

# Before training
testing(inputDataNormalized, weightArray, class_colors, fig, axs[0], 'Before training')

# Training
shuffledInputs = list(range(numberOfInputes))
np.random.shuffle(shuffledInputs)
for e in range(epoch):
    print(f"Epoch: {e + 1}/{epoch}")
    learningRate = CalculateLearningRate(initialLearningRate, learningDecayRate, e)
    width = CalculateFunctionWidth(initialFunctionWidth, widthDecay, e)
    for i in range(numberOfInputes):
        input = inputDataNormalized[i, :]
        winner = CalculateWinningNeuron(input, weightArray)
        winnerRow = winner[0]
        winnerCol = winner[1]

        for r in range(np.size(outputArray, 0)):
            for c in range(np.size(outputArray, 0)):
                p1 = np.array([r, c])
                p2 = np.array([winnerRow, winnerCol])
                distance = np.linalg.norm(p1 - p2) ** 2
                h = np.exp(-distance / (2 * width ** 2))
                deltaW = learningRate * h * (input - weightArray[r, c, :])
                weightArray[r, c, :] = weightArray[r, c, :] + deltaW

# After training
testing(inputDataNormalized, weightArray, class_colors, fig, axs[1], 'After training')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {round(elapsed_time)} seconds")

# Show the plot
plt.show()
