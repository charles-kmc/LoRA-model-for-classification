import torch.nn as nn
import torch

from tqdm import tqdm
import os


class Trainer(object):
    def __init__(self, model, train_loader, test_loader, lr = 1e-3, device = "cpu", save_ckeckpoint = 100, path_checkpoint = ""):
        self.device = device
        self.lr = lr
        self.save_ckeckpoint = save_ckeckpoint
        self.path_checkpoint = path_checkpoint
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    
    def training_loop(self, epochs, lora = False):
        
        
        print("\t ***** Training the model *****")
        for epoch in range(epochs):
            self.epoch = epoch
            train_loader_iterator = tqdm(self.train_loader, total = len(self.train_loader), desc = f"Epoch {epoch + 1}/{epochs}")
            train_avg_loss = self.train_epoch()
            train_loader_iterator.set_postfix(train_avg_loss = train_avg_loss)
            
            # test model
            test_loader_iterator = tqdm(self.test_loader, total = len(self.test_loader), desc = f"Epoch {epoch + 1}/{epochs}")
            test_avg_loss, accuracy = self.test_epoch()
            test_loader_iterator.set_postfix(test_loss = test_avg_loss, accuracy = accuracy)
            
            if epoch % self.save_ckeckpoint == 0:
                self.save_model(epoch, test_avg_loss, self.path_checkpoint,lora)
                
        self.save_model(epoch, test_avg_loss, self.path_checkpoint, lora)
        
        print("\t ***** Training completed *****")
            
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)
    
    def test_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        self.wrong_counts = [0 for i in range(10)]
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                
                # forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                # get predicted labels
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                
                # count correct predictions
                correct += (predicted == labels).sum().item()
                
                # count wrong predictions
                wrong = (predicted != labels).int()
                for ii, val in enumerate(wrong):
                    if val:
                        self.wrong_counts[labels[ii].item()] += 1
                    else:
                        pass
                        
        return running_loss / len(self.test_loader), correct / total
    
    def save_model(self, epoch, loss, path_model_save, lora = False):
        s_dir = os.path.join(path_model_save, "LoRa" if lora else "No_LoRa")
        os.makedirs(s_dir, exist_ok = True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'lr': self.lr,
            }, os.path.join(s_dir,  f"digt_classification_{epoch}.pt"))