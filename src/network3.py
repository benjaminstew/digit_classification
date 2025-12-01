'''
Network v3: Network v2 with L2 regularisation, cross-entropy cost, and improved weight initialisation.
- v3.1: added ability to independently test accuracy and cost on test/validation and training data
'''
import random 
import numpy as np 

'''miscellaneous global functions'''
def sigmoid(Z):
    return 1.0/(1.0+np.exp(-Z)) #applied elementwise 

def sigmoid_prime(Z):
    return sigmoid(Z)*(1-sigmoid(Z))

def vectorised_digit(digit):
    '''turn digit into (10, 1) unit vector representation'''
    e = np.zeros((10,1))
    e[digit] = 1.0
    return e 

class CrossEntropyCost():
    @staticmethod 
    def function(a, y):
        '''cost of single training example with output activation a and truth label y'''
        cost_vector = np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))
        return sum(cost_vector) 

    @staticmethod 
    def delta_L(Z, A, Y):
        '''error from output layer'''
        return (A-Y) 
    
class QuadraticCost():
    @staticmethod
    def function(a, y):
        '''cost of single training example with output activation a and truth label y'''
        cost_vector = 0.5*np.linalg.norm(a-y)**2
        return sum(cost_vector) 
    
    @staticmethod
    def delta_L(Z, A, Y):
        return (A-Y)*sigmoid_prime(Z)  

class Network():
    def __init__(self, sizes, mini_batch_size, cost=CrossEntropyCost): #cross-entropy cost set as default 
        self.num_layers = len(sizes)
        self.mini_batch_size = mini_batch_size
        self.sizes = sizes 
        self.cost = cost
        self.default_weight_initialiser() #non-standard Gaussian weights set as default 
    
    def default_weight_initialiser(self):
        self.biases = [np.random.randn(x,self.mini_batch_size) for x in self.sizes[1:]] #list of bias vectors for each layer 
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for y,x in zip(self.sizes[1:], self.sizes[:-1])] #list of weight matrices for each layer  

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, X):
        """Inference step. Return the output of the NN if "X" is input for entire mini-batch."""
        for B, w in zip(self.biases, self.weights): #loop through b and w of each layer in network 
            X = sigmoid(np.dot(w, X)+B) #calc activation for layer, move forward
        return X

    def SGD(self, training_data, num_epochs, eta, lmbda, eval_data=None, monitor_eval_cost=False, 
            monitor_eval_accuracy=False, monitor_training_cost=False, monitor_training_accuracy=False):
        """Train the NN using mini-batch stochastic gradient descent.  
        The "training_data" is a list of tuples "(x, y)" representing the training inputs and the desired binary vector outputs. 
        "eval_data" is test or validation data, a list of tuples "(x, y)" representing the training inputs and the desired 0->9 digit labels.
        We can monitor the cost and accuracy on either the eval data or the training data, by setting the appropriate flags.  
        This is useful for tracking progress, but slows things down substantially as it adds an inference step after each epoch of training."""
        if eval_data: 
            n_eval = len(eval_data)
            eval_cost, eval_accuracy = [], [] #array of costs/accuracies for each epoch
        n = len(training_data)
        training_cost, training_accuracy = [], []

        #print initial metrics on training data (random noise model)
        '''initial_cost = self.calculate_cost(training_data, n, lmbda, eval=True)
        initial_accuracy = self.calculate_accuracy(training_data, n, eval=True)
        print("Initial cost on training data: {:.2f}".format(initial_cost))
        print("Initial accuracy on training data: {} / {} ({:.2f}%)".format(initial_accuracy, n, ((initial_accuracy/n)*100)))'''

        for j in range(num_epochs): #loop through multiple epochs of training 
            random.shuffle(training_data) #random reordering of training examples 
            epoch = [training_data[k:k+self.mini_batch_size] for k in range(0, n, self.mini_batch_size)] #generate mini-batches for epoch 
            for mini_batch in epoch: #loop over each mini-batch 
                self.update_mini_batch(mini_batch, eta, lmbda, n) 
            
            print("\nEpoch {0} complete".format(j))
            if monitor_training_cost:
                cost = self.calculate_cost(training_data, n, lmbda, eval=True)
                training_cost.append(cost)
                print(" Training cost: {:.2f}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.calculate_accuracy(training_data, n, eval=True)
                training_accuracy.append(accuracy)
                print(" Training accuracy: {} / {} ({:.2f}%)".format(accuracy, n, ((accuracy/n)*100)))
            if monitor_eval_cost:
                cost = self.calculate_cost(eval_data, n_eval, lmbda)
                eval_cost.append(cost)
                print(" Evaluation cost: {:.2f}".format(j, cost))
            if monitor_eval_accuracy:
                accuracy = self.calculate_accuracy(eval_data, n_eval)
                eval_accuracy.append(accuracy)
                print(" Evaluation accuracy: {} / {} ({:.2f}%)".format(accuracy, n_eval, ((accuracy/n_eval)*100)))

        return eval_cost, eval_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the NN's weights and biases by applying
        stochastic gradient descent using backpropagation to an entire mini batch at once.
        The "mini_batch" is a list of tuples "(x, y)", "eta"
        is the learning rate, "lmbda" is regularisation factor, "n" is training set size."""
        nabla_B = [np.zeros(B.shape) for B in self.biases] #initialise nB matrices for each layer to zero vector 
        nabla_w = [np.zeros(w.shape) for w in self.weights] #initialise nw matrices for each layer to zero matrix 
        X = np.transpose([np.reshape(x, 784) for x,y in mini_batch]) #X is (784, mini_batch_size) matrix
        Y = np.transpose([np.reshape(y, 10) for x,y in mini_batch]) #Y is (10, mini_batch_size) matrix 
    
        delta_nabla_B, delta_nabla_w = self.backprop(X, Y) #perform backprop on entire mini-batch, return list containing b and w gradients of each layer
        nabla_B = [nB+dnB for nB, dnB in zip(nabla_B, delta_nabla_B)] #update nB and nw with B and w gradients for entire mini-batch, for each layer  
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    
        self.weights = [(1-eta*(lmbda/n))*w-(eta/self.mini_batch_size)*nw for w, nw in zip(self.weights, nabla_w)] #update weights and biases for epoch with nB and nw of mini-batch, for each layer. L2 reg included
        self.biases = [B-(eta/self.mini_batch_size)*nB for B, nB in zip(self.biases, nabla_B)]

    def backprop(self, X, Y):
        """Return a tuple (nabla_B, nabla_w) representing the
        gradient for the entire mini-batch's quadratic cost function C_x. nabla_B and
        nabla_w are layer-by-layer lists of numpy arrays."""
        nabla_B = [np.zeros(B.shape) for B in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        '''feedforward''' 
        activations = [X] 
        activation = X
        Zs = [] #list to store Z matrices of each layer 
        for w,B in zip(self.weights, self.biases):
            Z = np.dot(w, activation) + B
            Zs.append(Z)
            activation = sigmoid(Z)
            activations.append(activation)
        
        '''output layer error'''
        delta_L = self.cost.delta_L(Zs[-1], activations[-1], Y) 
        nabla_B[-1] = delta_L #compute the gradients for outer layer 
        nabla_w[-1] = np.dot(delta_L, activations[-2].transpose())

        '''backpropagation of error'''
        delta_l = delta_L 
        for l in range(2, self.num_layers):
            delta_l = np.dot(self.weights[-l+1].transpose(), delta_l) * sigmoid_prime(Zs[-l])
            nabla_B[-l] = delta_l #compute the gradients for layer
            nabla_w[-l] = np.dot(delta_l, activations[-(l+1)].transpose())

        return (nabla_B, nabla_w)
    
    def calculate_accuracy(self, data, n, eval=False):
        """After inference with "data", return the number of inputs in "data" for which the NN outputs the correct result. 
        Note that the NN's output is assumed to be the index of which ever neuron in the final layer has the highest activation.
        The flag "eval" should be set to False if the data set is validation or test data, and to True if the
        data set is the training data."""
        batched_data = [data[k:k+self.mini_batch_size] for k in range(0, n, self.mini_batch_size)]
        results = []  
        if eval:
            for batch in batched_data:
                X = np.transpose([np.reshape(x, 784) for x,y in batch]) #x is (784, 1) vector so X is (mini_batch_size, 784) matrix
                Y = np.transpose([np.reshape(y, 10) for x,y in batch]) #y is (10, 1) vector so Y is (10, mini_batch_size) matrix
                OUT = self.feedforward(X) #OUT is (10, mini_batch_size) matrix  
                for i in range(self.mini_batch_size):
                    results.append((np.argmax(OUT[:, i]), np.argmax(Y[:, i]))) #select columns representing output activation and truth label of each test image  
        else:
            for batch in batched_data:
                X = np.transpose([np.reshape(x, 784) for x,y in batch]) 
                Y = [int(y) for x,y in batch] #y is 0->9 digit 
                OUT = self.feedforward(X)
                for i in range(self.mini_batch_size):
                    results.append((np.argmax(OUT[:, i]), Y[i])) 
        
        return sum(int(x == y) for x,y in results)
    
    
    def calculate_cost(self, data, n, lmbda, eval=False):
        """Return the total cost after inference with "data". 
        The flag "eval" should be set to False if the data set is validation or test data, and to True if the
        data set is the training data."""
        batched_data = [data[k:k+self.mini_batch_size] for k in range(0, n, self.mini_batch_size)]
        cost = 0.0
        if eval:
            for batch in batched_data:
                X = np.transpose([np.reshape(x, 784) for x,y in batch]) 
                Y = np.transpose([np.reshape(y, 10) for x,y in batch]) 
                OUT = self.feedforward(X)
                for i in range(self.mini_batch_size): 
                    cost += self.cost.function(OUT[:, i], Y[:, i]) 
            cost = cost/n #multiply by normalisation term 
            cost += 0.5*(lmbda/n)*sum(np.linalg.norm(w)**2 for w in self.weights) #add L2 reg term 
        else:
            for batch in batched_data:
                X = np.transpose([np.reshape(x, 784) for x,y in batch]) 
                Y = np.transpose([np.reshape(vectorised_digit(y), 10) for x,y in batch]) #y is 0->9 digit so Y is (10, mini_batch_size) matrix 
                OUT = self.feedforward(X)
                for i in range(self.mini_batch_size): 
                    cost += self.cost.function(OUT[:, i], Y[:, i]) 
            cost = cost/n
            cost += 0.5*(lmbda/n)*sum(np.linalg.norm(w)**2 for w in self.weights)  
        
        return cost
     



