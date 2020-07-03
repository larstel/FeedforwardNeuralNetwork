from NeuralNetworkTools import NeuralNetworkTools

# datasets
inputs = [[0, 0], [1, 0], [0, 1], [1, 1]]
expected_output = [[0], [1], [1], [0]]

# parameters
epochs = 400  # 400
bias = 1
learning_rate = 0.5  # 0.5
activation_function = "sigmoid"
number_of_neurons = [2, 10, 1]  # input/hidden/output

# create a feedforward network
tools = NeuralNetworkTools(
    bias, learning_rate, activation_function, number_of_neurons, len(inputs)
)

# initialise weights
layer1_weights, layer2_weights = tools.initialise_weights()

# training
# --- start backpropagation algorithm
plot_values = []
for i in range(epochs):
    for j in range(len(inputs)):
        # ==== forward pass ====
        output_hidden_nodes, output_output_nodes, input_hidden_nodes = tools.forward_pass(
            inputs,
            bias,
            layer1_weights,
            layer2_weights,
            activation_function,
            j,
        )
        # ==== backward pass ====
        weight_change_layer2, weight_change_layer1 = tools.backward_pass(
            inputs,
            layer2_weights,
            output_hidden_nodes,
            output_output_nodes,
            input_hidden_nodes,
            learning_rate,
            activation_function,
            j,
            expected_output,
            bias,
        )
        # ==== update weights ====
        tools.update_weights(
            layer1_weights,
            layer2_weights,
            weight_change_layer1,
            weight_change_layer2,
        )

        # --- end backpropagation

        # get the last outputs for plotting
        if i >= epochs - 1:
            print(output_output_nodes)

tools.plot_error()
