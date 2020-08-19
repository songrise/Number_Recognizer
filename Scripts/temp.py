input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

with open("Model\\model.txt", "w+") as m:
    m.write(str([input_nodes, hidden_nodes,
                 output_nodes, learning_rate]))
