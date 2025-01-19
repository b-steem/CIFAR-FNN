# CIFAR-FNN

Trains an FNN model and a logistic regression model on the CIFAR-10 dataset.

## FNN Architecture

$Y_p = Softmax(Relu(Tanh(X*W_1+b_1)*W_2+b_2)*W_3+b_3)$

## USAGE

Clone the repo and run the following commands.

`python train_regression.py mode=logistic` to train the logistic regression model.

`python fnn.py mode=fnn` to train the fully connected neural network.

`python fnn.py mode=tune target_metric=accuracy` to compare the results of fnn and logistic regression.