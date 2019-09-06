import numpy as np

"""
Simple small neural network library.
Elegantly written in Python with Numpy.

Erik Quintanilla

"""


def sigmoid(x):
        return np.exp(x)/(np.exp(x) +1 )

def relu(x):
        return np.maximum(x, 0)

def softmax(x):
        return np.exp(x)/sum(np.exp(x))

def cross_entropy(y, y_hat):
        return np.dot(-y,np.log(y_hat))



class NN():
    def __init__(self, layers = []):
        self.layers = layers

    def forward(self, x):
        for i in range(len(self.layers)):
            if i==0:
                self.layers[i].forward(x)
            else:
                previous = self.layers[i-1].output
                self.layers[i].forward(previous)

            if i == len(self.layers)-1:
                print("Output Vector: ", self.layers[i].output)
                # print("Weights: ", self.layers[i].weights)

            # print(self.layers[i].weights)
            # print(self.layers[i].bias)
            # print(self.layers[i].output)

    def backward(self, inputs, ground_truth):
        for i in range(len(self.layers)-1, -1, -1):
            if i == len(self.layers)-1:
                # we should assume the output layer isn't going to be relu...
                y = ground_truth
                y_hat = self.layers[i].output
                print("Loss: %f" %(cross_entropy(y, y_hat)))
                dLdb = y_hat - y
                dLdW = np.outer(dLdb, self.layers[i-1].output)
                self.layers[i].bias -= dLdb
                self.layers[i].weights -= dLdW
                self.layers[i].dLdb = dLdb
                self.layers[i].dLdW = dLdW
            else:
                Z = self.layers[i].output
                W_next = self.layers[i+1].weights
                dLdb_next = self.layers[i+1].dLdb
                Z_prev = self.layers[i-1].output

                if self.layers[i].activation == "relu":
                     dLdb = (W_next.T@(dLdb_next))*(np.heaviside(Z, 0))
                else:
                    dLdb = (W_next.T@(dLdb_next))*Z*(1-Z)

                if i == 0:
                    dLdW = np.outer(dLdb, inputs)
                else:
                    dLdW = np.outer(dLdb, Z_prev)
                self.layers[i].bias -= dLdb
                self.layers[i].weights -= dLdW
                self.layers[i].dLdb = dLdb
                self.layers[i].dLdW = dLdW

    def print_weights(self):
        for layer in self.layers:
            print(layer.weights)

    def add_layer(self, layer):
        self.layers.append(layer)


class FCLayer():
    def __init__(self, num_nodes, weights, activation = "sigmoid"):
        self.weights = weights
        self.bias = None
        #self.bias = self.bias / (np.linalg.norm(self.bias)) #this should be uniform random noise
        self.activation = activation
        self.output = None
        self.num_nodes = num_nodes
        self.dLdb = None
        self.dLdW = None

    def forward(self, x):
        wx = np.matmul(self.weights, x)
        self.bias = np.zeros(wx.shape)
        wx = np.add(wx, self.bias)

        if self.activation == "sigmoid":
                wx = sigmoid(wx)

        elif self.activation == "relu":
                wx = relu(wx)

        elif self.activation == "softmax":
                wx = softmax(wx)

        self.output = wx

