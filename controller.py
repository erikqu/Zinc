from ann import *


net = NN()

weights1 = np.array([[0.1, 0.3, 0.7],[0.9, 0.4, 0.4]])
layer1 = FCLayer(num_nodes=3, weights = weights1, activation = "relu")
net.add_layer(layer1)


weights2 = np.array([[0.4, 0.3],[0.7, 0.2]])
layer2 = FCLayer(num_nodes=2, weights = weights2, activation = "sigmoid")
net.add_layer(layer2)


weights3 = np.array([[0.5, 0.6],[0.6, 0.7], [0.3, 0.2]])
layer3 = FCLayer(num_nodes=2, weights = weights3, activation = "softmax")
net.add_layer(layer3)


inputs = np.array([0.1, 0.7, 0.3])
ground_truth = np.array([1.,0.,0.])
n_epochs = 200
for x in range(1,n_epochs+1):
    print("EPOCH: %i/%i" %(x, n_epochs))
    net.forward(inputs)
    net.backward(inputs, ground_truth)

net.print_weights()









