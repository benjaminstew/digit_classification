# digit_classification
My NumPy-only implementations of shallow neural networks that learn to classify handwritten digits, the main project in Michael Neilsen's fantastic book _Neural Networks and Deep Learning_. It is trained on the MNIST dataset (70000 images in mnist.pkl.gz) split into a 50000-image training set, 10000-image validation set and 10000-image test set. I am using this book to gain an intuitive understanding of how neural nets learn. 

- network.py is a the chapter 1 implementation, a shallow neural net using mini-batch stochastic gradient descent with no optimisations.
  Example run on an Apple M2 of a three layer network with sizes [784, 30, 10], learning rate = 3.0, 30 epochs and mini batch size of 10 gives:  

  _% python3 training_run.py_   
  Epoch 0: 9042 / 10000 (90.42%)  
  Epoch 5: 9382 / 10000 (93.82000000000001%)  
  Epoch 10: 9427 / 10000 (94.27%)  
  Epoch 15: 9484 / 10000 (94.84%)  
  Epoch 20: 9481 / 10000 (94.81%)  
  Epoch 25: 9485 / 10000 (94.85%)   
  Epoch 29: 9477 / 10000 (94.77%)  
  Training took 79.958410042 seconds     

- network2.py is the chapter 2 implementation. It is the same as network.py but with batched backpropagation, so the gradients for all training examples in a mini-batch are computed simultaneously.
  Example run on an Apple M2 of a three layer network with sizes [784, 30, 10], learning rate = 3.0, 30 epochs and mini batch size of 10 gives:  

  _% python3 training_run.py_   
  Epoch 0: 9055 / 10000 (90.55%)  
  Epoch 5: 9392 / 10000 (93.92%)  
  Epoch 10: 9434 / 10000 (94.34%)  
  Epoch 15: 9461 / 10000 (94.61%)  
  Epoch 20: 9451 / 10000 (94.51%)  
  Epoch 25: 9488 / 10000 (94.88%)  
  Epoch 29: 9484 / 10000 (94.84%)  
  Training took 16.019150959 seconds

- network3.py is the chapter 3 implementation. Same as network2.py with improved weight initialisation, cross-entropy cost, and L2 regularisation.
  Example run on an Apple M2 of a three layer network with sizes [784, 30, 10], learning rate = 0.5, 30 epochs, regularisation factor = 5.0 and mini batch size of 10 gives:    

  _% python3 training_run.py_  
  Epoch 0: 9349 / 10000 (93.49%)   
  Epoch 5: 9542 / 10000 (95.42%)  
  Epoch 10: 9558 / 10000 (95.58%)  
  Epoch 15: 9574 / 10000 (95.74000000000001%)  
  Epoch 20: 9585 / 10000 (95.85000000000001%)  
  Epoch 25: 9595 / 10000 (95.95%)  
  Epoch 29: 9610 / 10000 (96.1%)  
  Training took 15.84347275 seconds

- I will next write code which: performs automatic tuning of the model's hyperparameters on the validation set (grid search, randomised search, Bayesian optimisation); plots accuracy vs #epochs to detect the extent of overfitting and inform an early stopping algorithm; and implements neuron dropout. I am building up to writing a CNN to solve the digit classification problem. 