import torch.nn as nn
import torch 
import os

dir = os.path.dirname(os.path.abspath(__file__))

from model import DigitClassification, LoRaParametrisation, LoRa_model, enable_disable_lora, count_parameters
from datasets import Datasets
from trainer import Trainer
from paths import *
import os

import warnings
import copy

# paths
save_dir = create_results_dir(dir_root = dir)
sub_dirs = ["checkpoints", "results"]
dic_pth = {}
for sub_ in sub_dirs:
    s_path = os.path.join(save_dir, sub_)
    dic_pth[sub_] = s_path
    os.makedirs(os.path.join(save_dir, sub_), exist_ok = True)
    

# device
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# load datasets
dataset = Datasets(dataset_name = "MNIST")
train_loader = dataset.get_dataloader(batch_size = 16)
test_loader = dataset.get_dataloader(batch_size = 16, train_status = False)

# Load model 
im_size = 28
classes = 10
model = DigitClassification(input_size = im_size*im_size, output_size = classes)

# Trainer
lr= 1e-3
save_ckeckpoint = 100 
path_checkpoint = dic_pth["checkpoints"]
trainer = Trainer(model, train_loader, test_loader, lr = lr, device = device, path_checkpoint=dic_pth["checkpoints"], save_ckeckpoint = save_ckeckpoint)

# Training
epochs = 1
trainer.training_loop(epochs = epochs, lora = False)


# copy weights and model
original_weights = {}
for name, param in model.named_parameters():
    original_weights[name] = param.data.clone().detach()
    
# Make a deep copy of the model
L_model = copy.deepcopy(model)

# Total number of paramters in the original model
print(f"Total trainable parameters of the model: {count_parameters(L_model)}") #type: ignore

# specific dataset to fine tuning the model
dataset_5 = Datasets(dataset_name = "MNIST", exclude_tgs=5)
train_loader_5 = dataset_5.get_dataloader(batch_size = 16)

# LoRa model
L_model = LoRa_model(L_model, rank = 2, device = device)

# Trainer for LoRa model
trainer_LoRa = Trainer(L_model, train_loader_5, test_loader, lr = lr, device = device, path_checkpoint=dic_pth["checkpoints"], save_ckeckpoint = save_ckeckpoint)

# training
trainer_LoRa.training_loop(epochs = 1, lora = True)

# test the model
enable_disable_lora(L_model, enabled=True)
trainer_LoRa.test_epoch()
for ii, w in enumerate(trainer_LoRa.wrong_counts):
    print(f"Misclassified {w} times for digit {ii}")

