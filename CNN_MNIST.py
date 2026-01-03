# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:03:48 2025

@author: Administrator
"""

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import time

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt


#%% CONFIG
epoch_count = 10
model_select = "LeNet5" # "Custom" or "LeNet5"

# For plotting purposes
train_accuracies = []
test_accuracies = []
start_time = time.time()
#%%


# LeNet-5 Implementation
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # padding to make 28x28 → 32x32
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool(x)
        x = F.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)  # raw scores (logits)
        return x

# Custom network 
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)  # shape: [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Define Train and Test functions
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        # print(f'data shape:{data.shape}')
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        preds = output.argmax(dim=1)
        total_correct += (preds == target).sum().item()
    return total_loss / len(train_loader.dataset), total_correct / len(train_loader.dataset)

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            total_correct += (preds == target).sum().item()
    return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)

#%%
def main():
  # Define transforms for CNN
  transform = transforms.Compose([
      transforms.Pad(2),  # pad to 32x32 (LeNet-5 expects 32x32)
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])

  train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
  test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True,num_workers=2)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False,num_workers=2)


  # Flatten images into [N, 784]
  X_train = train_dataset.data.view(-1, 28*28).float() / 255.0
  y_train = train_dataset.targets

  X_test = test_dataset.data.view(-1, 28*28).float() / 255.0
  y_test = test_dataset.targets


  # Loading MNIST Data as 2D + padded


  # Run Training
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # print(device)
  if model_select == "LeNet5":
    model = LeNet5().to(device) 
    print("LeNet5 model Selected")
  elif model_select == "Custom": 
    model = CustomCNN().to(device)
    print("Custom model Selected")
  else:
    model = LeNet5().to(device)
    print("Wrong model option - Selected LeNet5 by default") 
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  for epoch in range(epoch_count):
      train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
      test_loss, test_acc = test(model, test_loader, criterion, device)
      print(f"Epoch {epoch+1}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
      train_accuracies.append(train_acc)
      test_accuracies.append(test_acc)

  # Plot after training
  plt.plot(range(1, epoch_count + 1),train_accuracies, label='Train Accuracy')
  plt.plot(range(1, epoch_count + 1),test_accuracies, label='Test Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Training vs Test Accuracy')
  plt.xticks(range(1, epoch_count + 1))  # Set xticks to 1 through 10
  plt.legend()
  plt.grid(True)
  plt.savefig("lenet5_accuracy_plot_1e_2.png")  # Saves the plot to 'accuracy_plot.png'
  plt.show()


if __name__ == "__main__":
     main()
   

end_time = time.time()  # Record end time
elapsed_time = end_time - start_time  # Compute elapsed time

print(f"Elapsed time: {elapsed_time:.4f} seconds")