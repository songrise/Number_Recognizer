from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import numpy
# %matplotlib inline


def recognize_1_sample(Model: NeuralNetwork):
    """
    randomly show a pic in the test dataset and return the recognized number.
    """
    with open("Data\\mnist_test.csv", "r") as test_data_file:
        test_row = 9999
        # Randomly pick a test entry in the test data file
        test_data_list = test_data_file.readlines()
        test_index = numpy.random.randint(0, test_row, dtype='int')
        all_values = test_data_list[test_index].split(',')
        # convert into an image array
        image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = Model.query(inputs)
        result = numpy.argmax(outputs)
        # show the results
        print("AI says it is number {}.".format(result))
        plt.imshow(image_array, cmap="Greys", interpolation='None')
        plt.show()

        return result
