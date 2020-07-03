import copy
import random
import math
import matplotlib.pyplot as plt

error_values = []


class NeuralNetworkTools:
    def __init__(
        self,
        bias,
        learning_rate,
        activation_function,
        number_of_neurons,
        number_of_inputs,
    ):
        self.bias = bias
        self.learning_rate = learning_rate
        self.number_of_input_neurons = number_of_neurons[0]
        self.number_of_hidden_neurons = number_of_neurons[1]
        self.number_of_output_neurons = number_of_neurons[2]
        self.number_of_inputs = number_of_inputs

    def matrix_multiplication(self, x, y):
        return [
            [sum(a * b for a, b in zip(x_row, y_col)) for y_col in zip(*y)]
            for x_row in x
        ]

    def transpose(self, matrix):
        new_matrix = []
        length = len(matrix)
        for i in range(length):
            new_matrix.append([matrix[i]])
        return new_matrix

    def appends(self, matrix1, matrix2):
        matrix2.insert(0, matrix1[0])
        return matrix2

    def forward_pass(
        self, inputs, bias, layer1_weights, layer2_weights, activation_function, x
    ):
        # ====Forward Pass====
        # first layer
        input_hidden_neuron = self.matrix_multiplication(
            layer1_weights, self.appends([[bias]], self.transpose(inputs[x]))
        )
        if activation_function == "sigmoid":
            output_hidden_neuron = self.sigmoid(copy.deepcopy(input_hidden_neuron))
        else:
            output_hidden_neuron = self.sigmoid_with_minus(
                copy.deepcopy(input_hidden_neuron)
            )

        # second layer
        input_output_neuron = self.matrix_multiplication(
            layer2_weights, self.appends([[bias]], copy.deepcopy(output_hidden_neuron))
        )
        if activation_function == "sigmoid":
            output_output_neuron = self.sigmoid(input_output_neuron)
        else:
            output_output_neuron = self.sigmoid_with_minus(input_output_neuron)

        return output_hidden_neuron, output_output_neuron, input_hidden_neuron

    def backward_pass(
        self,
        inputs,
        layer2_weights,
        output_hidden_neuron,
        output_output_neuron,
        input_hidden_neuron,
        lr,
        activation_function,
        x,
        expected_output,
        bias,
    ):
        # ====Backward Pass====
        error = expected_output[x][0] - output_output_neuron[0][0]
        # print(str(expected_output[x][0]) + " - " + str(output_output_neuron[0][0]) + " = " + str(error))
        error_values.append(error)  # appends all errors for graphical representation

        # SECOND LAYER
        ###################################
        # Wj(t+1) = Wj(t) + 2*lr*error*Sj #
        ###################################
        weight_change_layer2 = self.matrix_multiplication(
            self.appends([[bias]], copy.deepcopy(output_hidden_neuron)),
            [[error * 2 * lr]],
        )

        # FIRST LAYER
        #######################################################
        # Wji(t+1) = Wji(t) + lr*2*(sum(error*Wj*Xi*Sj(1-Sj)) #
        #######################################################
        part2 = []
        weight_change_layer1 = []

        for i in range(len(output_hidden_neuron) * len(output_output_neuron)):
            if activation_function == "sigmoid":
                part1 = (
                    error
                    * layer2_weights[0][i + 1]
                    * self.sigmoid_derivation(input_hidden_neuron[i][0])
                )
            else:
                part1 = (
                    error
                    * layer2_weights[0][i + 1]
                    * self.sigmoid_derivation_with_minus(input_hidden_neuron[i][0])
                )

            # create the x vector
            x_vector = [[bias]]
            for j in range(len(inputs[0])):
                x_vector[0].append(inputs[x][j])

            part2 = self.matrix_multiplication([[part1]], x_vector)
            part3 = self.matrix_multiplication([[2 * lr]], part2)

            weight_change_layer1.append(part3[0])

        return weight_change_layer2, weight_change_layer1

    ##########################
    # mathematical functions #
    ##########################
    def sigmoid(self, x):
        for i in range(len(x)):
            x[i][0] = 1 / (1 + math.exp(-x[i][0]))
        return x

    def sigmoid_with_minus(self, x):
        for i in range(len(x)):
            x[i][0] = ((1 / (1 + math.exp(-x[i][0]))) * 2) - 1
        return x

    def sigmoid_derivation(self, x):
        return 1 / (1 + math.exp(-x)) * (1 - (1 / (1 + math.exp(-x))))

    def sigmoid_derivation_with_minus(self, x):
        return (2 * math.exp(-x)) / ((1 + math.exp(-x)) * (1 + math.exp(-x)))

    def initialise_weights(self):
        # initialise variables
        layer1_weights = []
        layer2_weights = []

        # ==== Layer 1 ====
        # append for every hidden neuron a weight matrix
        for _ in range(self.number_of_hidden_neurons):
            # one hidden neuron has weights from input neuron + bias
            layer1_weights.append(
                [
                    random.uniform(-1, 1)
                    for _ in range(0, self.number_of_input_neurons + 1)
                ]
            )

        # ==== Layer 2 ====
        for _ in range(self.number_of_output_neurons):
            # one output neuron has weights from hidden neuron + bias
            layer2_weights.append(
                [
                    random.uniform(-1, 1)
                    for _ in range(0, self.number_of_hidden_neurons + 1)
                ]
            )

        return layer1_weights, layer2_weights

    def update_weights(
        self, layer1_weights, layer2_weights, weight_change_layer1, weight_change_layer2
    ):
        # ==== update weights ====
        # second layer
        for k in range(len(layer2_weights[0])):
            layer2_weights[0][k] += weight_change_layer2[k][0]

        # first layer
        for l in range(len(layer1_weights)):
            for m in range(len(layer1_weights[0])):
                layer1_weights[l][m] += weight_change_layer1[l][m]

    def plot_error(self):
        tt = [abs(number) for number in error_values]
        temp = [sum(tt[i: i + self.number_of_inputs]) for i in range(0, len(tt), self.number_of_inputs)]

        plt.plot(temp)
        plt.title("Errorcurve")
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.show()
