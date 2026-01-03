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

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import time
start_time = time.time()

#%% CONFIG
Kernel = 'poly' #['linear', 'poly', 'rbf']
pca_dim = 80
lda_dim = 8 # max M-1 =9
dim_method = 'pca' # 'pca' or 'lda'

#%% Define Multiclass Logistic Regression

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)  # raw scores → CrossEntropyLoss applies softmax internally

#%%
def main():
    # Load Dataset 
    # Define transforms for SVM/Logistic Regression
    transform = transforms.Compose([
        transforms.ToTensor(),  # convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # normalize to mean/std
    ])
    
    # Download training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    
    ##########################################################
    
    # Flatten images into [N, 784]
    X_train = train_dataset.data.view(-1, 28*28).float() / 255.0
    y_train = train_dataset.targets
    
    X_test = test_dataset.data.view(-1, 28*28).float() / 255.0
    y_test = test_dataset.targets
    
    ###########################################################
    
    # Dimensionality Reduction
    # 1. PCA
    pca = PCA(n_components=pca_dim)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # 2. LDA
    lda = LinearDiscriminantAnalysis(n_components=lda_dim)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    
    ###########################################################
    # Original Data
    # X_train = X_train.clone().detach().float()
    # X_test = X_test.clone().detach().float()
    
    
    # Apply PCA
    if dim_method == 'pca':
        X_train = torch.tensor(X_train_pca).float()
        X_test = torch.tensor(X_test_pca).float()
    elif dim_method == 'lda':
    # Apply LDA
        X_train = torch.tensor(X_train_lda).float()
        X_test = torch.tensor(X_test_lda).float()
    else:
        print("select correct dim method, pca or lda")
    
    ###########################################################
    # 3. SVM training (example with linear kernel)
    clf = svm.SVC(kernel=Kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))
    
    
    ##########################################################################################
    
    model = LogisticRegression(input_dim=X_train.shape[1], num_classes=10)
    criterion = nn.CrossEntropyLoss()  # negative log-likelihood
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    # Training loop
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)  # shape [N, 10]
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
        # Optional: compute accuracy
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y_train).float().mean().item()
    
        # print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {acc*100:.2f}%")
    ##########################################################################################
    
    with torch.no_grad():
        outputs_test = model(X_test)
        preds_test = torch.argmax(outputs_test, dim=1)
        test_acc = (preds_test == y_test).float().mean().item()
    
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
if __name__ == "__main__":
     main()
   

end_time = time.time()  # Record end time
elapsed_time = end_time - start_time  # Compute elapsed time

print(f"Elapsed time: {elapsed_time:.4f} seconds")