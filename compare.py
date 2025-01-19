"""
Implements code for logistic regression, fully connected neural network, and hyperparameter search.

"""

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def logistic_regression(device):
    batch_size_train = 200
    batch_size_test = 1000

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # get the data
    train_loader, validation_loader, test_loader = get_data(batch_size_train, batch_size_test)

    # create the model
    multi_logistic_model = MultipleLogisticRegression().to(device)
    
    # parameters
    n_epochs = 10
    learning_rate = 5e-3
    momentum = 0.9

    l_type = "L1"
    # L1 regularization parameter
    l1_lambda = 0.0001

    # L2 regularization parameter
    weight_decay = 5e-4

    # define the optimizer
    if l_type == "L2":
        # weight decay L2 https://discuss.pytorch.org/t/how-to-add-a-l2-regularization-term-in-my-loss-function/17411/5 accessed 2024-09-2024
        optimizer = optim.SGD(multi_logistic_model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    else:
        optimizer = optim.SGD(multi_logistic_model.parameters(), lr=learning_rate, momentum=momentum)

    eval(validation_loader,multi_logistic_model,"Validation",device,l1_lambda,l_type)
    for epoch in range(1, n_epochs + 1):
        train(epoch,train_loader,multi_logistic_model,optimizer, device)
        eval(validation_loader,multi_logistic_model,"Validation",device,l1_lambda,l_type)
        
    eval(test_loader,multi_logistic_model,"Test",device,l1_lambda,l_type)

    for param in multi_logistic_model.parameters():
        print(param)

    results = dict(
        model=multi_logistic_model,
    )

    return results

def get_data(batch_size_train, batch_size_test):
     # create training, validation, and test sets
    MNIST_training = datasets.MNIST('/MNIST_dataset', train=True, download=True,transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))]))
    
    MNIST_test_set = datasets.MNIST('/MNIST_dataset/', train=False, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))]))
    
    MNIST_training_set, MNIST_validation_set = random_split(MNIST_training, [len(MNIST_training) - 12000, 12000])

    train_loader = DataLoader(MNIST_training_set,batch_size=batch_size_train, shuffle=True)

    validation_loader = DataLoader(MNIST_validation_set,batch_size=batch_size_train, shuffle=True)

    test_loader = DataLoader(MNIST_test_set,batch_size=batch_size_test, shuffle=True)

    return train_loader, validation_loader, test_loader

def train(epoch,data_loader,model,optimizer, device):
    log_interval = 100
    # define one hot encoding
    one_hot = F.one_hot

    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, one_hot(target, num_classes=10).float())

        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
        
def eval(data_loader, model, dataset, device, l1_lambda, l_type):
    # define one hot encoding
    one_hot = F.one_hot
    loss = 0
    correct = 0
    with torch.no_grad(): # notice the use of no_grad
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

            if l_type == "L1":
                # https://www.geeksforgeeks.org/l1l2-regularization-in-pytorch/ accessed 2024-09-24
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm

            else:
                loss += F.cross_entropy(output, one_hot(target, num_classes=10).float(), size_average=False).item() 
        
    loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    model.loss = loss
    model.accuracy = accuracy
    print(dataset+'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), accuracy))

class MultipleLogisticRegression(nn.Module):
    def __init__(self):
        super(MultipleLogisticRegression, self).__init__()
        self.fc = nn.Linear(28*28, 10)
        self.loss = None
        self.accuracy = None

    def forward(self, x):
        # https://machinelearningmastery.com/building-a-logistic-regression-classifier-in-pytorch/ accesed 2024-09-24
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()

        self.loss_type = loss_type
        self.num_classes = num_classes

        """add your code here"""
        # add the linear models
        self.fc1 = nn.Linear(32*32*3,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        """ 
        Takes a batch of images as a tensor of size N x (32*32*3) and returns the class probablilities as a tensor of size N x 10.
        
        Args:
            x: N x (32*32*3) tensor of batch images
        Returns:
            y: N x 10 tensor with the probablities of each image falling into each category
        """

        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x), dim=1)

        return output

    def get_loss(self, output, target):
        """
        Computes the loss according to the loss type argument of __init__
        
        Args:
            output: the output of the forward pass (predictions)
            target: the ground truth labels
        Returns:
            loss: the loss of the model"""
    
        if self.loss_type == "ce":
            loss = F.cross_entropy(output, target)

        else:
            raise NotImplementedError

        return loss


def tune_hyper_parameter(target_metric, device):
    """
    This function tunes the hyperparameters of part 1 and 2
    
    Args:
        target_metric: the metric to use (either accuracy or loss)
        device: the device to run the tuning on
    Returns:
        best_params: the best parameters the tuning found
        best_metric: the best metric the tuning found
    """
    # mapped as [parameter_name, [max_value, min_value, step_size]]
    parameters = [
        [7e-3, 7e-4, 5e-4],     # learning rate
        [0.99, 0.9, 0.01],      # momentum
        [2e-4, 5e-5, 5e-5],     # l2_weight
    ]

    # run search on logistic regression
    logistic_metric, logistic_params = random_search(target_metric, device, "logistic", parameters)

    parameters = [
        [5e-1, 5e-4, 5e-4],     # learning rate
        [0.99, 0.5, 0.05],      # momentum
    ]
    # run search on fnn
    fnn_metric, fnn_params = random_search(target_metric, device, "fnn", parameters)

    best_params = [logistic_params, fnn_params]
    best_metric = [
        {
            "logistic":logistic_metric
        },
        {
            "fnn":fnn_metric
        }

    ]
    return best_params, best_metric

    

def random_search(target_metric, device, model_type, parameters):
    """
    Runs a random search to find the parameters maximize the model on the target metric
    
    Args:
        target_metric: the metric to use (either accuracy or loss)
        device: the device to run the tuning on
        model: the model to run on (either logistic or fnn)
        parameters: a dictonary of the parameters to tune
        best_metric: the best metric value we have found (either in loss or accuraccy)
        
    Returns:
        best_metric: the best metric found so far
        best_params: the best parameters found
        """
    import random

    best_metric = 0. if target_metric == "accuracy" else 1.
    best_params = None

    num_iterations = 8

    if model_type == "logistic": 
        # run logistic  
        min_learning_rate = parameters[0][1]
        max_learing_rate = parameters[0][0]

        min_momentum = parameters[1][1]
        max_momentum = parameters[1][0]
                
        min_l2_weight = parameters[2][1]
        max_l2_weight = parameters[2][0]

        for _ in range(num_iterations):      
            
            learning_rate = round(random.uniform(min_learning_rate, max_learing_rate), 6)
            momentum = round(random.uniform(min_momentum, max_momentum), 6)
            l1_lambda = round(random.uniform(min_l2_weight, max_l2_weight), 6)
            
            # run the model on training set
            n_epochs = 5
            random_seed = 1
            torch.backends.cudnn.enabled = False
            torch.manual_seed(random_seed)

            # get the data
            train_loader, validation_loader, _ = get_data(200, 50)

            # create the model
            multi_logistic_model = MultipleLogisticRegression().to(device)

            # optimizer 
            optimizer = optim.SGD(multi_logistic_model.parameters(), lr=learning_rate, momentum=momentum)

            for epoch in range(1, n_epochs + 1):
                train(epoch,train_loader,multi_logistic_model,optimizer, device)
                eval(validation_loader,multi_logistic_model,"Validation",device,l1_lambda,"L1")

            metric_value = multi_logistic_model.accuracy

            if metric_value > best_metric:
                # update metric and params
                best_metric = metric_value
                best_params = [
                    {
                        "learning_rate":learning_rate,
                        "momentum":momentum,
                        "regularization_weight":l1_lambda,
                    }
                ]

                print(f"Current best accuracy: {best_metric}")
                print(f"Current best params: {best_params}")


    elif model_type == "fnn":  
        min_learning_rate = parameters[0][1]
        max_learing_rate = parameters[0][0]

        min_momentum = parameters[1][1]
        max_momentum = parameters[1][0]

        # run fnn tuning
        for _ in range(num_iterations):
            learning_rate = round(random.uniform(min_learning_rate, max_learing_rate), 6)
            momentum = round(random.uniform(min_momentum, max_momentum), 6)

            params = Params(learning_rate,momentum)

            train_loader, val_loader, _ = get_dataloaders(params.batch_size)


            net = FNN(params.loss_type, 10).to(device)
            optimizer = optim.SGD(net.parameters(), lr=params.learning_rate,
                                momentum=params.momentum)

            with torch.no_grad():
                validation_fnn(net, val_loader, device)
            for epoch in range(params.n_epochs):
                print(f'\nepoch {epoch + 1} / {params.n_epochs}\n')

                train_fnn(net, optimizer, train_loader, device)

                with torch.no_grad():
                    metric_value = validation_fnn(net, val_loader, device)

            if metric_value > best_metric:
                # update metric and params
                best_metric = metric_value
                best_params = [
                    {
                        "learning_rate":params.learning_rate,
                        "momentum":params.momentum,
                    }
                ]
                print(f"Current best accuracy: {best_metric}")
                print(f"Current best params: {best_params}")
        

    else:
        raise AssertionError(f"model: '{model_type}' is not one of logistic or fnn")

    print(f"Best metric: {best_metric}")
    print(f"Best params: {best_params}")

    return best_metric, best_params

"""FNN copy for part 3"""
class Params:
    class BatchSize:
        train = 128
        val = 128
        test = 100

    def __init__(self, learning_rate, momentum):
        self.mode = 'fnn'
        # self.model = 'tune'
        self.target_metric = 'accuracy'
        # self.target_metric = 'loss'

        self.device = 'gpu'
        self.loss_type = "ce"
        self.batch_size = Params.BatchSize()
        self.n_epochs = 5
        self.learning_rate = 1e-1 if learning_rate == None else learning_rate
        self.momentum = 0.5 if momentum == None else momentum


def get_dataloaders(batch_size):
    
    import torch
    from torch.utils.data import random_split
    import torchvision

    """

    :param Params.BatchSize batch_size:
    :return:
    """

    CIFAR_training = torchvision.datasets.CIFAR10('.', train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    CIFAR_test_set = torchvision.datasets.CIFAR10('.', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    # create a training and a validation set
    CIFAR_train_set, CIFAR_val_set = random_split(CIFAR_training, [40000, 10000])

    train_loader = torch.utils.data.DataLoader(CIFAR_train_set, batch_size=batch_size.train, shuffle=True)

    val_loader = torch.utils.data.DataLoader(CIFAR_val_set, batch_size=batch_size.val, shuffle= False)

    test_loader = torch.utils.data.DataLoader(CIFAR_test_set,
                                              batch_size=batch_size.test, shuffle= False)

    return train_loader, val_loader, test_loader


def train_fnn(net, optimizer, train_loader, device):
    from tqdm import tqdm
    net.train()
    pbar = tqdm(train_loader, ncols=100, position=0, leave=True)
    avg_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss = net.get_loss(output, target)
        loss.backward()
        optimizer.step()

        loss_sc = loss.item()

        avg_loss += (loss_sc - avg_loss) / (batch_idx + 1)

        pbar.set_description('train loss: {:.6f} avg loss: {:.6f}'.format(loss_sc, avg_loss))


def validation_fnn(net, validation_loader, device) -> float:
    net.eval()
    validation_loss = 0
    correct = 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss = net.get_loss(output, target)
        validation_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    validation_loss /= len(validation_loader.dataset)
    accuracy = 100. * correct / len(validation_loader.dataset)
    print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        validation_loss, correct, len(validation_loader.dataset),
        accuracy))
    
    return accuracy