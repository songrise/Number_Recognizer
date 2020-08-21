from NNsample import *
from NNtrain import *
import numpy
import matplotlib.pyplot
import scipy.special


def main():
    print("*Number Recognizer*")
    if input("Do you want to train the model on your machine? (Y/N):") in ["Y", "y"]:
        input_nodes = 784
        hidden_nodes = 200
        output_nodes = 10
        learning_rate = 0.1
        epochs = 5
        if input("Do you want to customize the model parameters? (Y/N):") in ["Y", "y"]:
            try:
                hidden_nodes = int(
                    input("Please input the number of hidden layer neurons(integer):"))
                if hidden_nodes <= 0:
                    raise ValueError
                learning_rate = float(
                    input("Please input the learning rate (0-1): "))
                if learning_rate <= 0 or learning_rate >= 1:
                    raise ValueError
                epochs = int(
                    input("please input the number of epochs(integer):"))
                if epochs < 1:
                    raise ValueError
            except ValueError:
                print("Invalid Input! Program Aborted.")
                exit(-1)

        train(input_nodes, hidden_nodes, output_nodes, learning_rate, epochs)

    # load the trained model data
    model_parameters = numpy.load("Model\\parameters.npy").tolist()
    # cast the data type of layer number into integer, while leaving the learning rate entry float
    model_parameters = list(
        map(int, model_parameters[:3])) + [model_parameters[-1]]
    model_wih = numpy.load("Model\\weight_ih.npy")
    model_who = numpy.load("Model\\weight_ho.npy")

    # set up the model, list unpack here
    model = NeuralNetwork(*model_parameters)
    # set the weight
    model.wih = model_wih
    model.who = model_who

    while 1:
        recognize_1_sample(model)


if __name__ == "__main__":
    main()
