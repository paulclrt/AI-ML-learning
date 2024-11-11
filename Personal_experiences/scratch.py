
import numpy as np






import numpy as np

class NeuralNet:
    def __init__(self, activation_func):
        self.activation_func = activation_func
        self.layers = []
        self.activations = []  # Store activations for each layer during forward pass

    def activation_function(self, x):
        if self.activation_func == "relu":
            return np.maximum(0, x)
        else:
            return np.maximum(0, x)  # Placeholder for other activations

    def activation_derivative(self, x):
        if self.activation_func == "relu":
            return np.where(x > 0, 1, 0)
        else:
            return 1  # Placeholder for other activation derivatives

    def loss_function(self, output, expected_output):
        return ((np.sum(output - expected_output)) ** 2) / 2

    def add_input_layer(self, input_values):
        n_input_nodes = len(input_values)
        self.layers.append({"n_of_nodes": n_input_nodes, "values": input_values})

    def add_hidden_layer(self, number_of_nodes):
        previous_number_of_nodes = self.layers[-1]['n_of_nodes']
        weights = np.random.rand(number_of_nodes, previous_number_of_nodes) / 100
        biases = np.random.rand(number_of_nodes) / 100
        self.layers.append({"n_of_nodes": number_of_nodes, "weights": weights, "biases": biases})

    def forward_prop(self):
        activation = self.layers[0]["values"]
        self.activations = [activation]  # Reset activations for a new pass

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            dot_prod = np.dot(activation, layer["weights"].T) + layer["biases"]
            activation = self.activation_function(dot_prod)
            self.activations.append(activation)

        return activation

    def backprop(self, expected_output, learning_rate=0.01):
        error = self.activations[-1] - expected_output
        
        for i in reversed(range(1, len(self.layers))):
            layer = self.layers[i]
            activation_prev = self.activations[i-1]
            derivative = self.activation_derivative(self.activations[i])

            dz = error * derivative
            dW = np.dot(dz[:, np.newaxis], activation_prev[np.newaxis, :])
            db = np.sum(dz, axis=0)

            layer["weights"] -= learning_rate * dW
            layer["biases"] -= learning_rate * db

            error = np.dot(dz, layer["weights"])

    def train(self, X, Y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            self.layers[0]["values"] = X  # Set input layer values
            output = self.forward_prop()
            self.backprop(Y, learning_rate)
            loss = self.loss_function(output, Y)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")


nn = NeuralNet(number_of_input=3, activation_func="relu")
nn.add_hidden_layer(4)
nn.add_hidden_layer(2)  # Output layer with 2 nodes (binary classification)

nn.train()