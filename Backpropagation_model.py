from random import random
from random import seed
from math import exp
from math import tanh


# Initialization of ANN with several hidden layers
def ini_network(n_inputs, n_hidden, n_outputs, added_sub_layers):
    network= list()
    hidden_layer_ini = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer_ini)
    for _ in range(added_sub_layers):
        hidden_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Calculate neuron activation for an input
def activate(weights, input):
    #Choosing final row of weights for the bias
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*input[i]
    return activation

# Transfer neuron activation
def transfer(activation, functiontype):
    if functiontype == 'Exponential':
        return 1.0 / (1.0 + exp(-activation))
    elif functiontype == 'Tanh':
        return tanh(activation)

# Forward propagation from input to a network output
def forward_propagation(network, instance, activation_function):
    inputs = instance
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation, activation_function)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output, activation_function):
    if activation_function == 'Exponential':
        return output * (1.0 - output)
    elif activation_function == 'Tanh':
        return 1 - (tanh(output))**2

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected, activation_function):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'], activation_function)


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        #input all weights except last term which is the bias
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs, activation_function):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagation(network, row, activation_function)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected, activation_function)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Make a prediction with a network
def predict(network, row, activation_function):
    outputs = forward_propagation(network, row, activation_function)
    return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation_sgs(train, test, l_rate, n_epoch, n_hidden, added_sub_layers):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = ini_network(n_inputs, n_hidden, n_outputs, added_sub_layers)
    train_network(network, train, l_rate, n_epoch, n_outputs, 'Exponential')
    predictions = list()
    for row in test:
        prediction = predict(network, row, 'Exponential')
        predictions.append(prediction)
    return(predictions)

