import numpy
import scipy.special
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    A neural network model with one hidden layer.
    """

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # node matrices
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        # Learning rate (alpha)
        self.lr = learningRate
        # link weight matrices wih(in-hidden) and who(hidden-out)
        self.wih = (numpy.random.normal(0.0, pow(self.inodes, - 0.5)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, - 0.5)))
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (
            1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * \
            numpy.dot((hidden_errors * hidden_outputs *
                       (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        # conver inputs list into a 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
