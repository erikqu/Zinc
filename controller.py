from zinc import NN, FCLayer
import numpy as np
from collections import Counter

input_shape = (2,)

net = NN(input_shape = input_shape, learning_rate = 0.5)

net.add_layer(FCLayer(num_nodes=input_shape[0],  activation = "sigmoid"))
net.add_layer(FCLayer(num_nodes=10, activation = "sigmoid"))
net.add_layer(FCLayer(num_nodes=1, activation = "sigmoid"))


net.compile()


def f(x):
    """
    Try and have the NN approximate the
    xor function.
    """
    if x[0] == x[1]:
        return 0.
    else:
        return 1.



datalength = 100
n_epochs = 200

xs = [np.random.randint(2, size=2) for x in range(0,datalength)]
ys = [[f(xs[x])] for x in range(len(xs))]

fake_xs = [tuple(x) for x in xs]

ctr = Counter(fake_xs)
print(ctr)

xs = np.asarray(xs)
ys = np.asarray(ys)


for x in range(1,n_epochs+1):
    for i in range(len(xs)):
        # print("INPUT", xs[i], xs[i].shape)
        # print("FORWARD PASS: ", i)
        net.forward(xs[i])
        net.backward(ys[i])
    print("EPOCH: %i/%i" %(x, n_epochs))

manned_input = input("TRAINING COMPLETE\n\nWOULD YOU LIKE TO ENTER INPUT MANUALLY? (y/n)>")

if manned_input.lower() == "y":
    while True:
        inp = np.array([])
        for x in range(input_shape[0]):
            tmp = input("Enter element # %i >>>>" %(x))
            inp = np.append(inp, float(tmp))
        net.forward(inp)
        response = net.layers[-1].output
        print(response)
