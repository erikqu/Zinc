import numpy as np
from numba import jit

"""
Small NN lib.

Erik Quintanilla

"""


@jit(nopython = True)
def sigmoid(x):
        return np.exp(x)/(np.exp(x) +1 )

@jit(nopython = True)
def relu(x):
        return np.maximum(x, 0)

@jit(nopython = True)
def softmax(x):
        return np.exp(x)/sum(np.exp(x))

def cross_entropy(y, y_hat):
        return -1*np.dot(y,np.log(y_hat))

def mean_squared_error(y, y_hat):
        return (1/(len(y)))*((y-y_hat)**2)

@jit(nopython = True)
def heaviside(x):
    return np.heaviside(0,x)


class NN():
    def __init__(self, input_shape, layers = [], learning_rate = 0.1):
        self.layers = layers
        self.cache = {}
        self.cache["input"] = None
        self.cache["gt"] = None
        self.input_shape = input_shape
        self.optimizer = None
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Calling forward on the network with an input will iterate
        over all the layers.
        """
        self.cache["input"] = x
        for i in range(len(self.layers)):
            if i==0:
                self.layers[i].forward(x)
            else:
                previous = self.layers[i-1].output # (x, y) * (y, 1)
                self.layers[i].forward(previous)

    def backward(self, ground_truth):
        """
        Compute a backward pass starting from the output layer
        and going backward. Printing the loss as we go and storing
        the output and gradients.
        """
        self.cache["gt"] = ground_truth
        for i in range(len(self.layers)-1, -1, -1):
            if i == len(self.layers)-1:
                y = ground_truth
                y_hat = self.layers[i].output
                print("Loss: %f" %(cross_entropy(y, y_hat)))
                dLdb = y_hat - y
                dLdW = np.outer(dLdb, self.layers[i-1].output)
                self.layers[i].bias -= self.learning_rate*dLdb
                self.layers[i].weights -= self.learning_rate*dLdW

                self.layers[i].dLdb = dLdb
                self.layers[i].dLdW = dLdW
            else:
                Z = self.layers[i].output
                W_next = self.layers[i+1].weights
                dLdb_next = self.layers[i+1].dLdb
                Z_prev = self.layers[i-1].output

                if self.layers[i].activation == "relu":
                     dLdb = (np.matmul(W_next.T, (dLdb_next)))*(np.heaviside(Z, 0))
                else:
                    dLdb = (np.matmul(W_next.T,(dLdb_next)))*Z*(1-Z)

                if i == 0:
                    dLdW = np.outer(dLdb, self.cache["input"])
                else:
                    dLdW = np.outer(dLdb, Z_prev)
                self.layers[i].bias -= self.learning_rate*dLdb
                self.layers[i].weights -= self.learning_rate*dLdW

                self.layers[i].dLdb = dLdb
                self.layers[i].dLdW = dLdW

    def compile(self):
        """
        When the weights and biases are not assigned
        we are to set them to random normally dist. noise.
        """
        for i in range(len(self.layers)):
            if self.layers[i].weights is None:
                if i == 0:
                    self.layers[i].weights = np.random.normal(0,1, (self.layers[i].num_nodes, self.input_shape[0]))
                else:
                    self.layers[i].weights = np.random.normal(0,1, (self.layers[i].num_nodes, self.layers[i-1].num_nodes))

            self.layers[i].bias = np.random.normal(0, 1, (self.layers[i].weights.shape[0], ))


    def print_weights(self):
        """
        Print out all the weight matrices of the network.
        """
        for layer in self.layers:
            print(layer.weights)

    def add_layer(self, layer):
        """
        Prettily add a layer to the network.
        """
        self.layers.append(layer)


class FCLayer():
    def __init__(self, num_nodes, weights = None, activation = "sigmoid"):
        self.weights = weights
        self.bias = None
        self.activation = activation
        self.output = None
        self.num_nodes = num_nodes
        self.dLdb = None
        self.dLdW = None

    def forward(self, x):
        """
        Computes the forward pass (activation(wx+b)) for a given layer
            x (tensor): input or output of previous layer
        """
        wx = np.matmul(self.weights, x)
        wx = np.add(wx, self.bias)

        if self.activation == "sigmoid":
                wx = sigmoid(wx)

        elif self.activation == "relu":
                wx = relu(wx)

        elif self.activation == "softmax":
                wx = softmax(wx)

        self.output = wx

