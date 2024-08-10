import torchvision.transforms as transforms #type: ignore
import torchvision.datasets as datasets #type: ignore
import torch
import os

class Datasets(object):
    def __init__(self, transform = None, dataset_name = "MNIST", exclude_tgs = None) -> None:
        self.transform = transform
        self.dataset_name = dataset_name
        self.exclude_tgs = exclude_tgs
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])   
        self.root = os.path.join("/home/cmk2000/Documents/Years 3/Python codes/Codes","Datasets")
        os.makedirs(self.root, exist_ok = True)
        
    def get_dataloader(self, batch_size, train_status = True):
        if self.dataset_name == "MNIST":
            self.dataset = datasets.MNIST(root = self.root, train = train_status, download = True, transform = self.transform)
        elif self.dataset_name == "CIFAR10":
            self.dataset = datasets.CIFAR10(root = self.root, train = train_status, download = True, transform = self.transform)
        
        if self.exclude_tgs is not None:
            exclude_indices = self.dataset.targets == self.exclude_tgs
            self.dataset.data = self.dataset.data[exclude_indices]
            self.dataset.targets = self.dataset.targets[exclude_indices]
        else:
            pass
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = batch_size, shuffle = True)
        
        return dataloader
        
        
        