from ann import *


input_shape = (2,)

net = NN(input_shape = input_shape)

layer1 = FCLayer(num_nodes=input_shape[0],  activation = "sigmoid")
net.add_layer(layer1)

layer2 = FCLayer(num_nodes=50, activation = "sigmoid")
net.add_layer(layer2)

layer3 = FCLayer(num_nodes=1, activation = "sigmoid")
net.add_layer(layer3)

net.compile()


def f(x):
    """
    Try and have the NN approximate the
    xor function.
    """
    if x[0] == x[1]:
        return 0
    else:
        return 1


xs = [np.random.randint(2, size=2) for x in range(0,200)]
ys = [[f(xs[x])] for x in range(len(xs))]

xs = np.asarray(xs)
ys = np.asarray(ys)


n_epochs = 1
for x in range(1,n_epochs+1):
    for i in range(len(xs)):
        # print("INPUT", xs[i], xs[i].shape)
        net.forward(xs[i])
        net.backward(ys[i])
    print("EPOCH: %i/%i" %(x, n_epochs))

while True:
    inp = np.array([])
    for x in range(input_shape[0]):
        tmp = input("Enter element # %i >>>>" %(x))
        inp = np.append(inp, float(tmp))
    net.forward(inp)
    response = net.layers[-1].output
    print(response)
