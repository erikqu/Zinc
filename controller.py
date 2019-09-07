from ann import *


net = NN(input_shape = (3,) )

layer1 = FCLayer(num_nodes=3,  activation = "relu")
net.add_layer(layer1)


layer2 = FCLayer(num_nodes=2, activation = "sigmoid")
net.add_layer(layer2)


layer3 = FCLayer(num_nodes=2, activation = "softmax")
net.add_layer(layer3)

net.compile()


inputs = np.array([0.7, 0.3, 0.0]) # np.random.normal(0,3, 3)
ground_truth = np.array([1,0,0])
n_epochs = 200
for x in range(1,n_epochs+1):
    print("EPOCH: %i/%i" %(x, n_epochs))
    net.forward(inputs)
    net.backward(ground_truth)

#net.print_weights()









