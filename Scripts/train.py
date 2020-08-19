import numpy
import matplotlib.pyplot as plt
import scipy.special
from neural_network import NeuralNetwork


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
model = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

######Training#######
with open("Data\\mnist_train.csv", "r") as train_data_file:
    training_data_list = train_data_file.readlines()
    epochs = 1
    print("Training Started")
    for e in range(epochs):
        # go through all records in the training data set
        print("Epoch {} ({} Remaining)".format(
            str(e+1), str(epochs - e - 1)))
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
    print("Training Finished")


########Testing#####
with open("Data\\mnist_test.csv", "r") as test_data_file:
    test_data_list = test_data_file.readlines()
    # test the neural network

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    print("Testing model")
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
    print("Error rate for the model (estimated): {:.4f} %".format(
          100*(1.0 - (scorecard_array.sum() / scorecard_array.size))))

    # Save the trained model
    with open("Model\\model.txt", "w+") as m:
        # NN infos
        m.write(str([model.inodes, model.hnodes,
                     model.onodes, model.lr])+"\n")
        # weight matrices
        weight_ih = model.wih.tolist()
        weight_ho = model.who.tolist()
        m.write(str(weight_ih)+"\n")
        m.write(str(weight_ho)+"\n")


input("Press any key to continue")
