# -*- coding: utf-8 -*-
"""
Created on Fri May  2 22:36:08 2025

@author: Administrator
"""
import torch
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights , VGG16_Weights
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import json


epoch_count = 10
epochs = list(range(1, epoch_count + 1))
model_select = "VGG" # "VGG" or "Resnet18"
train_option = "fine_tune" # "last_FC_only" or "fine_tune"

train_dataset_path = '/content/drive/MyDrive/ENEE633/project2/archive/training/training'
val_dataset_path = '/content/drive/MyDrive/ENEE633/project2/archive/validation/validation'

#%%
# For plotting purposes
simple_val_accuracies = []
simple_train_accuracies = []
transfer_val_accuracies = []

start_time = time.time()

# Define Simple CNN 
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Define Train and Test functions
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
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

  train_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
  ])
  val_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
  ])

  train_dataset = datasets.ImageFolder(train_dataset_path, transform=train_transform)
  val_dataset = datasets.ImageFolder(val_dataset_path, transform=val_transform)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=2)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False,num_workers=2)

  
  # Testing Simple Model
  print("Testing simple model")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
  model = SimpleCNN().to(device)  
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

  for epoch in range(epoch_count):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, val_loader, criterion, device)
    simple_train_accuracies.append(train_acc)
    simple_val_accuracies.append(test_acc)
    print(f"Epoch {epoch+1}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

  # Plot for SimpleCNN
  plt.figure()
  plt.plot(epochs, simple_train_accuracies, label='SimpleCNN Train Accuracy')
  plt.plot(epochs, simple_val_accuracies, label='SimpleCNN Val Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Training & Validation Accuracy - SimpleCNN ')
  plt.xticks(epochs)
  plt.grid(True)
  plt.legend()
  plt.savefig("simple_val_accuracy.png")
  plt.close()
  
  ###############################################################################################################################
  #%% Transfer Learning using ResNet or VGG

  # Freezing and Training last FC layer / Fine Tuning
  # Load pretrained model
  if model_select == "Resnet18": 
    print("Resnet18 model Selected")
    # Resnet18
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Freeze all layers / or leave as is
    for param in model.parameters():
      if train_option == "last_FC_only":
        param.requires_grad = False # For only training last layer
      elif train_option == "fine_tune":
        param.requires_grad = True # For Fine tuning
      else:
        param.requires_grad = False # For only training last layer
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

  elif model_select == "VGG":
    print("VGG16 model Selected")
    # VGG model
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    # Freeze all convolutional layers
    for param in model.parameters():
      if train_option == "last_FC_only":
        param.requires_grad = False # For only training last layer
      elif train_option == "fine_tune":
        param.requires_grad = True # For Fine tuning
      else:
        param.requires_grad = True # For only training last layer
    # Replace final classifier layer
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 10)  # 10 monkey classes
  else:
    print("Wrong model option")

  # Select the optimizer
  if train_option == "last_FC_only" and model_select == "Resnet18":
    print("learning rate - 0.0005")
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0005) # Only train fc
  elif train_option == "last_FC_only" and model_select == "VGG":
    print("learning rate - 0.0003")
    optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=0.0003) # Only train fc    
  elif train_option == "fine_tune":
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # For fine-tuning
    print("learning rate - 0.00001")
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # Only train fc 

  # print train option
  if train_option == "last_FC_only":
    print("training last FC layer ...")
  elif train_option == "fine_tune":
    print("Fine tuning ...")
  else:
    print("Wrong training option - fine tuning by default")


  # Loss function
  criterion = nn.CrossEntropyLoss()

  # Move to GPU if available
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  #print(device)
  # Train loop for only last FC layer
  for epoch in range(epoch_count):
      train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
      val_loss, val_acc = test(model, val_loader, criterion, device)
      transfer_val_accuracies.append(val_acc)
      print(f"Epoch {epoch+1}, Val Acc: {val_acc:.4f}")


  # Plotting accuracies 
  # Plot validation accuracy for both models
  

  

  # Plot for Transfer Learning model
  plt.figure()
  plt.plot(epochs, transfer_val_accuracies, label=f'{model_select} Val Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title(f'Validation Accuracy - {model_select} ({train_option})')
  plt.xticks(epochs)
  plt.grid(True)
  plt.legend()
  plot_filename = f"val_accuracy_{model_select}_{train_option}.png"
  plt.savefig(plot_filename)
  plt.close()

  # Save transfer validation accuracies
  val_acc_filename = f"val_acc_{model_select}_{train_option}.json"
  with open(val_acc_filename, 'w') as f:
    json.dump(transfer_val_accuracies, f)


if __name__ == "__main__":
     main()

end_time = time.time()  # Record end time
elapsed_time = end_time - start_time  # Compute elapsed time

print(f"Elapsed time: {elapsed_time:.4f} seconds")