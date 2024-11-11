
import numpy as np






class NeuralNet():

    def __init__(self, number_of_input, input_values, activation_func):
        # no input validation for now because this is just to understand the basics, not deploy
        self.number_of_input = number_of_input # number of nodes in input layer
        self.activation_func = activation_func
        self.layers = [{"n_of_nodes": number_of_input, "values": input_values}]



    def activation_function(self, x):
        if self.activation_func == "relu":
            return np.maximum(0, x)
        else:
            return np.maximum(0, x) # for now just because too lazy to implement another

    def loss_function(self, output, expected_output):
        loss = ((np.sum(output-expected_output))**2)/2
        return loss


    def add_hidden_layer(self, number_of_nodes):
        """take the last layer number of nodes. make a matrice of size (number_of_nodes, number_of_previous_node)"""
        if len(self.layers) == 1:
            # only input layer has been added so ge the previous number of nodes from self.numer_of_ninput
            previous_number_of_nodes = self.number_of_input
        else:
            #get the layer object number of nodes
            previous_number_of_nodes = self.layers[-1]['n_of_nodes']

        weights = np.random.randint(100, size=(number_of_nodes, previous_number_of_nodes))/1000
        biases = np.random.randint(100, size=(number_of_nodes, ))/100
        self.layers.append({"n_of_nodes": number_of_nodes, "weights": weights, "biases": biases})


    def forward_prop(self):

        activation = self.layers[0]["values"]

        # Propagate through each layer
        for i in range(1, len(self.layers)):  # Skip input layer, start from first hidden layer
            layer = self.layers[i]

            # Perform dot product (input * weights) + biases
            dot_prod = np.dot(activation, layer["weights"].T) + layer["biases"].T

            # Apply activation function (ReLU)
            activation = self.activation_function(dot_prod)

        return activation  # This will be the output of the network
    




nn = NeuralNet(number_of_input=3, input_values=np.array([1, 0, 1]), activation_func="relu")
nn.add_hidden_layer(4)
nn.add_hidden_layer(2)  # Output layer with 2 nodes (binary classification)

output = nn.forward_prop()  # Perform forward propagation
print("Output:", output)