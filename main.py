from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(16, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    epoch_loss = 0
    correct = 0
    test_num = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        
        # compute accuracy
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        test_num += len(data)
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))
        # keep track of epoch loss
        epoch_loss += loss.item() * output.shape[0]

    train_accuracy = 100. * correct / test_num
    train_loss = epoch_loss/len(train_loader.sampler)
    print("epoch {}: training accuracy {:.2f}".format(epoch,train_accuracy))
    
    return train_accuracy, train_loss


def test(model, device, loader, set_name, return_labels=False):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in loader:
            # save ground truth for later
            true_labels.extend(target.numpy())
            # actual prediction
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)
            # save preds for later
            preds = pred.detach().cpu().numpy()
            pred_labels.extend(preds.flatten())

    test_loss /= test_num
    test_accuracy = 100. * correct / test_num

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        set_name, test_loss, correct, test_num,
        test_accuracy))
    
    if return_labels:
        return test_accuracy, test_loss, true_labels, pred_labels 
    
    return test_accuracy, test_loss


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=80, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='add data augmentation')
    parser.add_argument('--fraction', type=int, default=1,
                        help='fraction of data to use for training (1/x)')
    
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file name')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--model-name', type=str, default='',
                        help='string to attach to saved model filename')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    
    # define transforms to apply to data
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    rot_trans_shear = transforms.RandomAffine(
        degrees=20,
        #translate=(0.10, 0.10),
        scale=None,
        #shear=[-20, 20, -20, 20],
        fill=0)
    
    # transforms
    augmented_transform = transforms.Compose([rot_trans_shear, basic_transform])
    transform = basic_transform
    
    trainval_dataset = datasets.MNIST('./data', train=True, download=True,
                transform=transform)
    augmented_trainval_dataset = datasets.MNIST('./data', train=True, download=True,
                transform=augmented_transform)
        

    # Pytorch has default MNIST dataloader which loads data at each iteration
    trainval_dataset = datasets.MNIST('./data', train=True, download=True,
                transform=transform)

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
#     subset_indices_train = range(len(train_dataset))
#     subset_indices_valid = range(len(train_dataset))

    # Train-Val Split (85/15%) -- stratified across classes
    trainval_size = len(trainval_dataset)
    train_size, val_size = int(trainval_size*0.85), int(trainval_size*0.15)
    
    # use only a fraction of training data?
    if args.fraction != 1:
        train_size  = int(train_size/args.fraction)
        val_size = trainval_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,
                                                       [train_size, val_size],
                                                       generator=torch.Generator().manual_seed(args.seed))
    if args.augment:
        augmented_train_dataset, _ = torch.utils.data.random_split(augmented_trainval_dataset,
                                                       [train_size, val_size],
                                                       generator=torch.Generator().manual_seed(args.seed))
    
    # loaders -- careful tranforms have been applied at the dataset level already
    #            and we don't want to augment the validation set
    if args.augment:
        train_loader = torch.utils.data.DataLoader(
            augmented_train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size,
        shuffle=True)

    # Load your model [fcNet, ConvNet, Net]
    model = ConvNet().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    epochs = range(1, args.epochs + 1)
    train_losses, val_losses = [0 for _ in epochs], [0 for _ in epochs]
    train_accuracies, val_accuracies = [0 for _ in epochs], [0 for _ in epochs]
    for i, epoch in enumerate(epochs):
        train_accuracies[i], train_losses[i] = train(args, model, device, train_loader, optimizer, epoch)
        val_accuracies[i], val_losses[i] = test(model, device, val_loader, "Validation")
        scheduler.step()    # learning rate scheduler
        
    print(train_losses)
    print(val_losses)
    
    # plot losses
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.xticks(epochs)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(labels=['train loss', 'val loss'])
    if args.augment:
        plt.title('losses with augmentation')
    plt.show()
    
    print("train accuracies:", train_accuracies)
    print("val accuracies:", val_accuracies)
    
    # plot accuracies
    plt.plot(epochs, train_accuracies)
    plt.plot(epochs, val_accuracies)
    plt.xticks(epochs)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(labels=['train accuracy', 'val accuracy'])
    if args.augment:
        plt.title('accuracies with augmentation')
    plt.show()

        # You may optionally save your model at each epoch here

    if args.save_model:
        if args.model_name:
            model_filename = "mnist_model_{}.pt".format(args.model_name)
        else:
            model_filename = "mnist_model.pt"
        # Verify that the folder where to save models exists
        if not os.path.isdir('./models'):
            os.makedirs('./models')
        model_path = os.path.join('./models', model_filename)
        print('Saving model to {}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        
    # Evaluate on the official test set
    if args.evaluate:
        # No augmentation when evaluating / testing
        test_dataset = datasets.MNIST('./data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test_accuracy, test_loss = test(model, device, test_loader, "Test")
        
        print("test loss:", test_loss)
        print("test accuracies:", test_accuracy)

if __name__ == '__main__':
    main()
