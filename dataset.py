import torch
import torchvision
from torchvision import datasets
import torch.utils.data as data


def get_mnist_dataset(data_dir, batch_size_train = 64, batch_size_test=1000):

    #TODO later: add validation option

    transform_train = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)) # global mean and standard deviation of the MNIST dataset
                             ])
    
    transform_test = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
    
    train_set = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader





