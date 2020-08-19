from neural_network import NeuralNetwork
import matplotlib.pyplot as plt

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
