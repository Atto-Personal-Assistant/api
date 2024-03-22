import math

from app.interactors.matrix_interactor import Matrix


def sigmoid(x: int, i, j):
    return 1 / (1 + math.exp(-x))


def derivative_sigmoid(x: int, i, j):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self,
                 input_nodes: int,
                 hidden_nodes: int,
                 output_nodes: int,
                 learning_rate: int = 0.1,):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.bias_input_to_hidden = Matrix(self.hidden_nodes, 1)
        self.bias_input_to_hidden.randomize()
        self.bias_hidden_to_output = Matrix(self.output_nodes, 1)
        self.bias_hidden_to_output.randomize()

        self.weights_input_to_hidden = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_input_to_hidden.randomize()

        self.weights_hidden_to_output = Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_hidden_to_output.randomize()

        self.learning_rate = learning_rate

    def train(self, input_arr, input_arr_expected):
        input_matrix = Matrix.array_to_matrix(input_arr)

        # INPUT TO HIDDEN #
        hidden = Matrix.multiply(self.weights_input_to_hidden, input_matrix)
        hidden = Matrix.add(hidden, self.bias_input_to_hidden)
        hidden.map(sigmoid)

        # HIDDEN TO OUTPUT #
        output = Matrix.multiply(self.weights_hidden_to_output, hidden)
        output = Matrix.add(output, self.bias_hidden_to_output)
        output.map(sigmoid)

        # BACK PROPAGATION #

        # OUTPUT -> HIDDEN #
        expected = Matrix.array_to_matrix(input_arr_expected)
        output_error = Matrix.subtract(expected, output)
        derivative_output = Matrix.static_map(output, derivative_sigmoid)

        hidden_transpose = Matrix.transpose(hidden)

        gradient_output = Matrix.hadamard(output_error, derivative_output)
        gradient_output = Matrix.scaled_multiply(gradient_output, self.learning_rate)

        # Adjust Bias
        self.bias_hidden_to_output = Matrix.add(self.bias_hidden_to_output, gradient_output)

        # Adjust Weights O -> H
        weights_hidden_to_output_deltas = Matrix.multiply(gradient_output, hidden_transpose)
        self.weights_hidden_to_output = Matrix.add(
            self.weights_hidden_to_output,
            weights_hidden_to_output_deltas,
        )

        # HIDDEN -> INPUT #
        weights_hidden_to_output_transposed = Matrix.transpose(
            self.weights_hidden_to_output,
        )
        hidden_error = Matrix.multiply(
            weights_hidden_to_output_transposed,
            output_error,
        )
        derivative_hidden = Matrix.static_map(hidden, derivative_sigmoid)
        input_transposed = Matrix.transpose(input_matrix)

        gradient_hidden = Matrix.hadamard(hidden_error, derivative_hidden)
        gradient_hidden.scaled_multiply(gradient_hidden, self.learning_rate)

        # Adjust Bias
        self.bias_input_to_hidden = Matrix.add(self.bias_input_to_hidden, gradient_hidden)

        # Adjust Weights H -> I
        weights_input_to_hidden_deltas = Matrix.multiply(gradient_hidden, input_transposed)
        self.weights_input_to_hidden = Matrix.add(
            self.weights_input_to_hidden,
            weights_input_to_hidden_deltas,
        )

    def predict(self, input_arr):
        input_matrix = Matrix.array_to_matrix(input_arr)

        # INPUT TO HIDDEN #
        hidden = Matrix.multiply(self.weights_input_to_hidden, input_matrix)
        hidden = Matrix.add(hidden, self.bias_input_to_hidden)
        hidden.map(sigmoid)

        # HIDDEN TO OUTPUT #
        output = Matrix.multiply(self.weights_hidden_to_output, hidden)
        output = Matrix.add(output, self.bias_hidden_to_output)
        output.map(sigmoid)

        output = Matrix.matrix_to_array(output)

        return output
