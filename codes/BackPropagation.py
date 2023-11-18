from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(5000)


class Neuron:
    def __init__(self, n_weights):
        self.weights = np.fromiter((np.random.random() for i in range(n_weights)), np.float32, n_weights)
        self.bias = np.random.random()
        self.n_weights = n_weights

    def sigmoid(self, value): # Activation function
        return 1 / (1 + np.exp(-value))

    def sigmoid_derivative(self, value): # Derivative of sigmoid
        return value * (1 - value)

    def activate(self, input): # Updating output using activation function
        self.output = self.sigmoid(np.dot(self.weights, input) + self.bias)

    def update_delta(self, error): # Updating delta using derivative of activation function
    	self.delta = error * self.sigmoid_derivative(self.output)


class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, l_rate):
    	self.layers = tuple(
            tuple(Neuron(n_inputs) for i in range(n_inputs)) # Hidden layer
            for n in range(n_hidden)) + ((Neuron(n_inputs),),) # Output layer
    	self.l_rate = l_rate
    	self.n_inputs = n_inputs
    	self.n_hidden = n_hidden

    def fit(self, train, n_epoch):
    	for epoch in range(n_epoch):
            for row in train:
                self.FP(row[:-1])
                self.BPE(row[-1])
                self.update_weights(row[:-1])
            print ('Step #: ',epoch)

    def FP(self, input): # Forward Propagate
        input = input.copy()
        for layer in self.layers:
            for i in range(len(layer)):
                layer[i].activate(input)
            for i in range(len(layer)):
                input[i] = layer[i].output

    def BPE(self, target): # Back Propagate Error
    	# Updating delta for each neuron
    	self.layers[-1][0].update_delta(target - self.layers[-1][0].output)
    	for i in reversed(range(self.n_hidden)):
    		for j in range(len(self.layers[i])):
    			self.layers[i][j].update_delta(sum(
    				neuron.weights[j] * neuron.delta
    				for neuron in self.layers[i + 1]))

    def update_weights(self, input): #Updating weights
        input = input.copy()
        for i in range(len(self.layers)):
            if i: # If current layer is not input layer
                for j in range(self.n_inputs): # Output become input
                    input[j] = self.layers[i - 1][j].output
            for neuron in self.layers[i]:
                neuron.weights += self.l_rate * neuron.delta * input
                neuron.bias += self.l_rate * neuron.delta # Updating bias

    def predict_sample(self, row): # Predict for one row
    	self.FP(row[:-1])
    	return self.layers[-1][0].output

    def predict(self, test): # Predict for test set
    	return np.fromiter((self.predict_sample(row) for row in test), np.float32, len(test))

    def error(self, target, predicted):
    	return 100 * np.abs(predicted - target).sum() / target.sum()


dataset = pd.read_csv('A1turbine.txt', sep=' ', skiprows=4, names=(
    'height over sea level', 'fall', 'net fall', 'flow', 'power'))
for column in dataset.columns:
    min = dataset[column].min()
    delta = dataset[column].max() - min
    dataset[column] = (dataset[column] - min) / delta
dataset = dataset.astype(np.float32).values

n_inputs = dataset.shape[1] - 1
l_rate = 0.3
n_epoch = 50000
n_hidden = 3

BP = NeuralNetwork(n_inputs, n_hidden, l_rate)
train, test = train_test_split(dataset, test_size=50)
BP.fit(train, n_epoch)
predictions = BP.predict(test)
target = test[:, -1]
total_error= round(BP.error(target, predictions), 2)

print('Error: {total_error}%',total_error)
print("delta", delta)
print("predictions", predictions)


predictions = predictions * delta
target = target * delta

plt.scatter(predictions, target, c='black', s=20)
plt.xlabel('Prediction')
plt.ylabel('Real')
plt.title('Prediction versus real')
plt.show()

with open('result.txt', 'w') as file:
    file.write('Predicted Target Error\n')
    for i in range(len(target)):
        file.write(f'{predictions[i]} {target[i]} {abs(target[i] - predictions[i])}')
        file.write(f'\nWeights: {[[neuron.weights.tolist() for neuron in layer] for layer in BP.layers]}\n')
        file.write(f'\nTotal error: {total_error}%\n')

# Cross-validation
k_folds = 11
folds = np.array(np.split(dataset, k_folds)) # Split array into k folds
errors = []
for k, fold in enumerate(folds):
    BP = NeuralNetwork(n_inputs, n_hidden, l_rate)
    print ('Fold #: ',k)
    BP.fit(np.delete(folds, k, axis=0).reshape(-1, n_inputs + 1), n_epoch)
    predictions = BP.predict(fold)
    target = fold[:, -1]
    errors.append(BP.error(target, predictions))

print(f'Cross-validation error: {round(sum(errors) / len(errors), 2)}%')
