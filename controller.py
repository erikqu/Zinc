from ann import *


net = NN()

# bias1 = np.ones((2,1))
# this bias is automatically set to the unit vector within
# the code.
weights1 = [[0.1, 0.3, 0.7],[0.9, 0.4, 0.4]]
weights1 = np.asarray(weights1)
layer1 = FCLayer(num_nodes=3, weights = weights1, activation = "relu")
net.add_layer(layer1)


#bias2 = np.ones((2,1))
weights2 = [[0.4, 0.3],[0.7, 0.2]]
weights2 = np.asarray(weights2)
layer2 = FCLayer(num_nodes=2, weights = weights2, activation = "sigmoid")
net.add_layer(layer2)


#bias3 = np.ones((2,1))
weights3 = [[0.5, 0.6],[0.6, 0.7], [0.3, 0.2]]
weights3 = np.asarray(weights3)
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









