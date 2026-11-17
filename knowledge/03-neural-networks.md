# Neural Networks: The Building Blocks

## Neurons and layers
An artificial neural network is made of simple units called neurons arranged in layers. Each neuron takes some numbers in, multiplies each by a weight, adds them up with a bias, and passes the result through an activation function such as ReLU, which simply turns negative numbers into zero. The output of one layer becomes the input of the next.

## Weights are the knowledge
The weights are the adjustable numbers inside the network. A large language model can have billions of them. Everything the model "knows" is stored implicitly in these weights. There is no database of facts inside; knowledge is spread across many weights in a way that is hard to read directly.

## Forward pass
Running an input through all the layers to get an output is called a forward pass. For a language model, the input is a sequence of tokens and the output is a probability for every possible next token.

## Why depth helps
Early layers tend to detect simple patterns, while deeper layers combine them into more abstract concepts. In image networks, early layers find edges and later layers find faces. In language models, early layers handle spelling and grammar while later layers handle meaning and reasoning-like patterns.

## Parameters vs. hyperparameters
Parameters (weights and biases) are learned during training. Hyperparameters, such as the number of layers, the learning rate, or the batch size, are chosen by humans before training starts.
