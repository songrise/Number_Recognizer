import numpy
import matplotlib.pyplot as plt
import scipy.special


class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # node matrices
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        # Learning rate (alpha)
        self.lr = learningRate
        # link weight matrices wih(in-hidden) and who(hidden-out)
        self.wih = numpy.random.normal(
            0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
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


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
model = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

######Training#######
with open("data\\mnist_train.csv") as train_data_file:
    training_data_list = train_data_file.readlines()
    epochs = 5
    for e in range(epochs):
        # go through all records in the training data set
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            model.train(inputs, targets)


########Testing#####
with open("data\\mnist_test.csv") as test_data_file:
    test_data_list = test_data_file.readlines()
    # test the neural network

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = model.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
    scorecard_array = numpy.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)
    while 1:
        i = int(input("Enter a index for test data and view the result:"))
        all_values = test_data_list[i].split(',')
        image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
        plt.imshow(image_array, cmap="Greys", interpolation='None')
        plt.show()
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = model.query(inputs)
        result = numpy.argmax(outputs)
        print("This should be number "+str(result))
        end = (True if input("Continue?(Y/N)") == "N" else False)
        if end:
            break


input("Press any key to continue")
