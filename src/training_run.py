import mnist_loader
import network3
#import network2
#import network
from time import perf_counter

start = perf_counter()
training_data, validation_data, test_data = mnist_loader.process_data()
mini_batch_size = 10
#net = network.Network([784, 30, 10]) 
#net = network2.Network([784, 30, 10], mini_batch_size)
net = network3.Network([784, 30, 10], mini_batch_size)
#net.SGD(training_data, 30, mini_batch_size, 3.0, test_data) 
eval_cost, eval_accuracy, training_cost, training_accuracy = net.SGD(training_data, 30, 0.5, 5.0, test_data, monitor_training_accuracy=True)
#all arrays should be 1D with length = num epochs 
end = perf_counter()
print("Training took {:.2f} seconds".format(end-start))

