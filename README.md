# Zinc


> Einstein's brain had a lot of zinc.

Simple neural network library written in Python using NumPy with syntax inspired by PyTorch. The library allows arbitrarily deep neural networks with sigmoid, relu, or softmax activations. 

## Usage: 
-   Inputs should be NumPy tensors
-   See example below for adding layers
-   Model layers must be sequential (for now) 
-   `net.forward(x)` does a forward pass on the input NumPy tensor `x`
-   `net.backward(y)` does a backward pass on the ground truth  `y`
-   You can run the example within the example folder with `python3 xor.py`




## Example: Approximating the XOR Function 
Walkthrough of the code in the example code given in this 
repo.

#### Using this network within controller.py:
```python
 input_shape = (2,)                                                        
 net = NN(input_shape = input_shape)                                             
 net.add_layer(FCLayer(num_nodes=input_shape[0],  activation = "sigmoid"))  
 net.add_layer(FCLayer(num_nodes=2, activation = "sigmoid"))              
 net.add_layer(FCLayer(num_nodes=1, activation = "sigmoid"))
 ```

(After training on the xor data...)

```
TRAINING COMPLETE

WOULD YOU LIKE TO ENTER INPUT MANUALLY? (y/n)>y
Enter element # 0 >>>>1
Enter element # 1 >>>>0
[0.99813859]
Enter element # 0 >>>>1
Enter element # 1 >>>>1
[0.00031161]
Enter element # 0 >>>>0
Enter element # 1 >>>>0
[0.0044377]
Enter element # 0 >>>>0
Enter element # 1 >>>>1
[0.99773632]
```


## TODO 

- Add evaluate function
- Add Convolutional layers 
- Add GPU support 
- Model zoo 
