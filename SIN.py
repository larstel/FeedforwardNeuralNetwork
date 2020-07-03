import numpy  # for plotting and sin input
import matplotlib.pyplot as plt  # for plotting
from NeuralNetworkTools import NeuralNetworkTools  # tools for backpropagation and feed

# datasets
inputs = []
expected_output = []
for i in range(0, 70):
    inputs.append([i * 0.1])  # input from 0 to 7 in 0.1 steps

    # sin output from 0 to 7 in 0.1 steps
    expected_output.append([numpy.sin(i * 0.1)])

print(inputs)

# parameters
epochs = 2000  # 500 # overfitting 100
bias = 1
learning_rate = 0.01  # 0.01
activation_function = "sigmoid_with_minus"
number_of_neurons = [1, 10, 1]  # input/hidden/output # 1,10,1 # overfitting 300
activate_gui = False

# create a feedforward network
tools = NeuralNetworkTools(
    bias,
    learning_rate,
    activation_function,
    number_of_neurons,
    len(inputs))

# initialise weights
layer1_weights, layer2_weights = tools.initialise_weights()

# --- start backpropagation algorithm
plot_values = []

# plotting the sinus
x = numpy.arange(0, 2 * numpy.pi, 0.1)   # start,stop,step
y = numpy.sin(x)
plt.title('Sinus')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.ion()

for i in range(epochs):
    for j in range(len(inputs)):
        # ==== forward pass ====
        output_hidden_nodes, output_output_nodes, input_hidden_nodes = tools.forward_pass(inputs, bias, layer1_weights, layer2_weights, activation_function, j)
        # ==== backward pass ====
        weight_change_layer2, weight_change_layer1 = tools.backward_pass(inputs, layer2_weights, output_hidden_nodes, output_output_nodes, input_hidden_nodes, learning_rate, activation_function, j, expected_output, bias)#weight_change_layer2, weight_change_layer1 = 
        # ==== update weights ====
        tools.update_weights(layer1_weights, layer2_weights, weight_change_layer1, weight_change_layer2)
# --- end backpropagation

        plot_values.append(output_output_nodes[0][0])

    plt.clf()
    plt.ion()
    plt.plot(x, y, label="SIN") 
    plt.plot(numpy.arange(0, 7, 0.1), plot_values, label="learned SIN (Dataset 1)")
    plt.legend()
    text = "epoch: " + str(i)
    plt.text(5, 0.8, text)
    plt.pause(0.00001)
    plot_values = []

# create 2nd Dataset
inputs = []
plot_values = []
for i in range(0, 70):
    inputs.append([i * 0.1 + 0.05])  # input from 0 to 7 in 0.1 steps

print()
print(inputs)
for j in range(len(inputs)):
    # forward pass for 2nd Dataset
    output_hidden_nodes, output_output_nodes, input_hidden_nodes = tools.forward_pass(inputs, bias, layer1_weights, layer2_weights, activation_function, j)
    plot_values.append(output_output_nodes[0][0])

plt.ion()
plt.plot(numpy.arange(0, 7, 0.1), plot_values, label="Forwardpass SIN (Dataset 2)")
plt.legend()
plt.pause(0.00001)

plt.ioff()
plt.show()

tools.plot_error()
