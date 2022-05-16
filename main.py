import numpy as np
import random
import time

class layer:
    def __init__(self, weights, biases, activation):
        self.weights = weights
        self.biases = biases
        self.activation = activation
    def forward(self, inputs):
        outputs = np.dot(inputs, np.array(self.weights).T) + self.biases
        return self.activation(outputs)

def randomWeights(numberOfInputs, numberOfNeurons):
    weights = []
    for i in range(numberOfNeurons):
        neuron = []
        for i in range(numberOfInputs):
            neuron.append((random.random() * 0.6) - 0.3) 
        weights.append(neuron)
    return weights


sigmoid = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
linear = np.vectorize(lambda x: x)
relu = np.vectorize(lambda x: max(0., x))
categoricalCrossEntropy = lambda prediction, target: -np.dot(target, np.log(np.clip(prediction, 1e-7, 1-1e-7)))

def softmax(values):
    outputs = values
    for i in range(len(outputs)):
        outputs[i] -= max(outputs[i])
    outputs = np.exp(outputs)
    for i in range(len(outputs)):
        outputs[i] /= np.sum(outputs[i])
    return outputs


def calculateLoss(outputs, targets):
    losses = []
    for i in range(len(outputs)):
        losses.append(categoricalCrossEntropy(outputs[i], targets[i]))
    return np.mean(losses)

def calculateRoughAccuracy(outputs, targets):
    predictions = np.argmax(outputs, axis=1)
    classTargets = np.argmax(targets, axis=1)
    accuracy = np.mean(predictions == classTargets)
    return accuracy

hiddenLayer1 = layer(randomWeights(3, 4), [0 for i in range(4)], relu)
hiddenLayer2 = layer(randomWeights(4, 4), [0 for i in range(4)], relu)
outputLayer = layer(randomWeights(4, 2), [0], softmax)

print(calculateRoughAccuracy([[0.4, 0.6], [1, 0]], [[0, 1], [1, 0]]))
print(calculateLoss([[1, 0], [0, 1]], [[0, 1], [1, 0]]))