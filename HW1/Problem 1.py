# # # # # # # # # # # # # # # # # # # # # # #
# Homework 1                                #
# Problem 1: Boolean functions              #
# Author: Marcus Hansen                     #
# Course: FFR135 Artificial Neural Networks #
# # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import sys
import random

np.set_printoptions(threshold=sys.maxsize)


# Function for creating all boolean functions for desired n
def all_Boolean_Functions(n, sampels):
    rows = 2 ** n

    columns = sampels

    BooleanFunctions = np.zeros([rows, columns])
    for row in range(rows):
        for col in range(columns):
            BooleanFunctions[row][col] = random.choice([-1, 1])

    # Removes duplicates
    BooleanFunctions = np.unique(BooleanFunctions, axis=-1)

    return BooleanFunctions


# Function for computing the patterns for boolean function
def compute_Patterns(n):
    rows = 2 ** n
    patterns = np.zeros([rows, n])

    for r in range(rows):
        for j in range(n):
            patterns[r][j] = (r >> (n - 1 - j)) & 1

    return patterns


# Function to initialize the weight vector
def initial_WeightVector(variance, size) -> np.array:
    w = []
    for i in range(size):
        sampel = np.random.normal(loc=0.0, scale=variance)
        w.append(sampel)
    return w


# Function for determining convergence
def test_program(weightVector, theta, booleanFunctions, booleanPattern, index, shufflePatternsIndex):
    count = 0
    for k in shufflePatternsIndex:
        inputPattern = booleanPattern[k]
        target = booleanFunctions[k, index]
        localField = np.dot(weightVector, inputPattern) - theta
        output = np.sign(localField)
        if output == 0:
            output = 1
        if (target - output) != 0:
            count += 1
    return count


# Main program
def main_program(n, epoches, eta):
    booleanFunctions = all_Boolean_Functions(n, 10 ** 4)

    # Boolean pattern
    booleanPattern = compute_Patterns(n)

    shufflePatternsIndex = list(range(np.size(booleanPattern, 0)))

    indexOfSampels = list(range(0, np.size(booleanFunctions, 1)))

    numberOfLinearlySeperable = 0
    for j in indexOfSampels:
        weightVector = initial_WeightVector(1 / np.sqrt(n), np.size(booleanPattern, 1))
        theta = 0
        for epoch in range(epoches):
            random.shuffle(shufflePatternsIndex)
            for k in shufflePatternsIndex:
                inputPattern = booleanPattern[k]
                target = booleanFunctions[k, j]
                localField = np.dot(weightVector, inputPattern) - theta
                output = np.sign(localField)
                if output == 0:
                    output = 1
                updateW = np.dot(eta * (target - output), inputPattern)
                theta = theta - eta * (target - output)
                weightVector = weightVector + updateW
        random.shuffle(shufflePatternsIndex)
        linearSeperable = test_program(weightVector, theta, booleanFunctions, booleanPattern, j, shufflePatternsIndex)
        if linearSeperable == 0:
            numberOfLinearlySeperable += 1

    return numberOfLinearlySeperable


main_program(2, 20, 0.05)
