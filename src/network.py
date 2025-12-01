'''
Network v1: Shallow neural network for handwritten digit classification, trained on MNIST dataset. 
The code and model are not optimised, and omit many desireable features.
'''
import random 
import numpy as np 

class Network():

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x,1) for x in sizes[1:]] #list containing bias vectors for each layer 
        self.weights = [np.random.randn(y,x) for y,x in zip(sizes[1:], sizes[:-1])] #list of weight matrices for each layer 
        #randomly initialised weights and biases 

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z)) #automatically applied elementwise if z a matrix 

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for x,y in test_data]
        return sum(int(x == y) for x,y in test_results)

    def feedforward(self, x):
        """Return the output of the whole network if "x" is input"""
        for b, w in zip(self.biases, self.weights): #iterate through b and w of each layer in network 
            x = self.sigmoid(np.dot(w, x)+b) #calc activation for layer, move forward
        return x

    def SGD(self, training_data, num_epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs. If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out. This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        for j in range(num_epochs): #loop through multiple epochs of training 
            random.shuffle(training_data) #random reordering of training examples 
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] #generate mini-batches for epoch 
            for mini_batch in mini_batches: #loop over each mini-batch 
                self.update_mini_batch(mini_batch, eta) 
            if test_data:
                eval = self.evaluate(test_data)
                print("Epoch {0}: {1} / {2} ({3}%)".format(j, eval, n_test, ((eval/n_test)*100)))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #initialise nb sum vectors for each layer to zero vector (sum = over all training examples in mini-batch)
        nabla_w = [np.zeros(w.shape) for w in self.weights] #initialise nw sum matrices for each layer to zero matrix 
        
        for x, y in mini_batch: #loop over all training examples in mini batch 
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #perform backprop on training example, return list containing b and w gradients of each layer
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #update nb and nw sums with b and w gradients for that training example, for each layer  
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] #update weights and biases for entire epoch with nb and nw of mini-batch, for each layer 
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives dC_x/da for the output activation vector"""
        return (output_activations - y) #quadratic cost function

    def backprop(self, x, y):
        """Return a tuple (nabla_b, nabla_w) representing the
        gradient for the training example's quadratic cost function C_x. nabla_b and
        nabla_w are layer-by-layer lists of numpy arrays, similar
        to self.biases and self.weights."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        '''feedforward''' 
        activations = [x] 
        activation = x
        zs = [] #list to store z vectors of each layer 
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        '''output layer'''
        delta_L = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1]) 
        nabla_b[-1] = delta_L #compute the gradients 
        nabla_w[-1] = np.dot(delta_L, activations[-2].transpose())

        '''backpropagation'''
        delta_l = delta_L 
        for l in range(2, self.num_layers):
            delta_l = np.dot(self.weights[-l+1].transpose(), delta_l) * self.sigmoid_prime(zs[-l])
            nabla_b[-l] = delta_l #compute the gradients 
            nabla_w[-l] = np.dot(delta_l, activations[-(l+1)].transpose())
        
        return (nabla_b, nabla_w)




        










    