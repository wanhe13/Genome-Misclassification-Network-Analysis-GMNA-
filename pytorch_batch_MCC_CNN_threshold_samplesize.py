#!/usr/bin/env python
# coding: utf-8

# # filter: only regions with genome counts>=threshold will be considered
# 
# # Sample: sample n_sample from each region
# 
# # Preprocessing: filter -> Sample -> padded -> encode -> leave one out
# 

import sys
print(sys.prefix)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
if torch.cuda.is_available():
    print("GPU is available.")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory Total (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
    print("GPU Memory Allocated (GB):", torch.cuda.memory_allocated(0) / 1e9)
    print("GPU Memory Cached (GB):", torch.cuda.memory_reserved(0) / 1e9)
else:
    print("GPU is not available.")


print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import vstack, csr_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import tensorflow as tf
from tensorflow.keras import layers

from utility_functions import *
from functions import *



#batch_size=128
#batch_size=256
batch_size=64

lr=0.00001
threshold = sys.argv[1]
sample_size = sys.argv[2]


data=pd.read_csv(f'data/genomeACTGbases_sample_{threshold}_{sample_size}.csv')  




    

class GenomeDataset(Dataset):
    def __init__(self, sequences, labels, base_to_index, label_to_index=None):
        self.sequences = sequences
        self.labels = labels
        self.base_to_index = base_to_index
        self.label_to_index = label_to_index or self._generate_label_to_index()

    def _generate_label_to_index(self):
        # Automatically generate a label-to-index mapping if not provided
        unique_labels = sorted(set(self.labels))  # Sort labels to ensure consistency
        return {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoded_sequence = self.one_hot_encode(sequence)
        # Convert labels to integers if necessary
        label_index = self.label_to_index[label] if isinstance(label, str) else label
        return encoded_sequence, torch.tensor(label_index, dtype=torch.long)

    def one_hot_encode(self, sequence):
        encoded = torch.zeros((len(sequence), len(self.base_to_index)), dtype=torch.float32)
        for i, base in enumerate(sequence):
            if base in self.base_to_index:
                encoded[i, self.base_to_index[base]] = 1
        return encoded




target='region'

bases = {'*', '-', 'A', 'C', 'G', 'T'}
base_to_index = {base: i for i, base in enumerate(sorted(bases))}


X_train, X_temp, y_train, y_temp = train_test_split(data['sequence'], data[target], test_size=0.4, random_state=42) 
X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)




train_dataset = GenomeDataset(X_train.tolist(), y_train.tolist(), base_to_index)
test_dataset = GenomeDataset(X_test.tolist(), y_test.tolist(), base_to_index)
validation_dataset = GenomeDataset(X_validation.tolist(), y_validation.tolist(), base_to_index)



train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)
validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False, num_workers=2)






n_class=len(list(set(list(data['region']))))






    
    
    
class CNNModelV5(nn.Module):
    def __init__(self, input_channels, n_filter, n_class):
        super(CNNModelV5, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, n_filter, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(n_filter)
        self.conv2 = nn.Conv1d(n_filter, n_filter, kernel_size=4)
        self.bn2 = nn.BatchNorm1d(n_filter)
        self.conv3 = nn.Conv1d(n_filter, n_filter, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(n_filter)
        self.conv4 = nn.Conv1d(n_filter, n_filter, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(n_filter)
        self.conv5 = nn.Conv1d(n_filter, n_filter, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(n_filter)    
        self.conv6 = nn.Conv1d(n_filter, n_filter, kernel_size=3)
        self.bn6 = nn.BatchNorm1d(n_filter) 
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(n_filter, n_class)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  
        x = F.relu(self.bn5(self.conv5(x))) 
        x = F.relu(self.bn6(self.conv6(x))) 
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x    
      

    
class CNNModelV6(nn.Module):
    def __init__(self, input_channels, n_filter, n_class):
        super(CNNModelV6, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, n_filter, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(n_filter)
        self.conv2 = nn.Conv1d(n_filter, n_filter, kernel_size=4)
        self.bn2 = nn.BatchNorm1d(n_filter)
        self.conv3 = nn.Conv1d(n_filter, n_filter, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(n_filter)
        self.conv4 = nn.Conv1d(n_filter, n_filter, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(n_filter)
        self.conv5 = nn.Conv1d(n_filter, n_filter, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(n_filter)    
        self.conv6 = nn.Conv1d(n_filter, n_filter, kernel_size=3)
        self.bn6 = nn.BatchNorm1d(n_filter) 
        self.conv7 = nn.Conv1d(n_filter, n_filter, kernel_size=3)
        self.bn7 = nn.BatchNorm1d(n_filter)         
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(n_filter, n_class)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  
        x = F.relu(self.bn5(self.conv5(x))) 
        x = F.relu(self.bn6(self.conv6(x))) 
        x = F.relu(self.bn6(self.conv7(x))) 
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x    
                
        
input_channels = 6
n_filter = 64
cnn_model = CNNModelV5(input_channels, n_filter=n_filter, n_class=n_class)


    
class CNNModel(nn.Module):
    def __init__(self, cnn_model):
        super(CNNModel, self).__init__()
        self.cnn_model = cnn_model

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn_model(x)

        return x


model = CNNModel(cnn_model)

   
if torch.cuda.is_available():
    print("GPU is available.")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory Total (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
    print("GPU Memory Allocated (GB):", torch.cuda.memory_allocated(0) / 1e9)
    print("GPU Memory Cached (GB):", torch.cuda.memory_reserved(0) / 1e9)
else:
    print("GPU is not available.")

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")




model.to(device)




model_name=f'CNNv5_{threshold}_{sample_size}'




output_filename = f'accuracy/{model_name}_{n_filter}/MCC_{model_name}_accuracy.txt'

print(output_filename)









# Create the directory for the output file if it doesn't exist
output_dir = os.path.dirname(output_filename)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


    
# List to store log messages for each epoch

def calculate_accuracy(y_pred, y_true):
    predicted_classes = torch.argmax(y_pred, dim=1)
    correct_predictions = (predicted_classes == y_true).float()
    accuracy = correct_predictions.sum() / len(y_true)
    return accuracy

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
print('lr',lr)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

num_epochs = 500

accumulation_steps = 4



if not os.path.exists(f"models/MCC_{model_name}_{n_filter}/"):
    os.makedirs(f"models/MCC_{model_name}_{n_filter}/")    




last_epoch = 0
for filename in os.listdir(f"models/MCC_{model_name}_{n_filter}/"):
    if filename.startswith(f'MCC_{model_name}') and filename.endswith('.pth'):
        epoch_number = int(filename.split('epoch')[1].split('.pth')[0])
        last_epoch = max(last_epoch, epoch_number)

checkpoint_path = f'models/MCC_{model_name}_{n_filter}/MCC_{model_name}_epoch{last_epoch}.pth'


# Load model if checkpoint exists
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    

    with open(output_filename, 'a') as f:
        f.write(f"Checkpoint found. Resuming training from epoch {start_epoch}\n")
        print(f"Checkpoint found. Resuming training from epoch {start_epoch}\n")
else:
    start_epoch = 0
    print(f"No checkpoint found. Start from scratch\n")

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        correct_predictions += (torch.argmax(outputs, dim=1) == labels).sum().item()
        total_samples += labels.size(0)

        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()

    # Validation loop
    model.eval()
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            val_running_loss += val_loss.item()

            val_correct_predictions += (torch.argmax(outputs, dim=1) == labels).sum().item()
            val_total_samples += labels.size(0)

    # Calculate and write average training and validation loss and accuracy to the output file
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / total_samples
    avg_val_loss = val_running_loss / len(validation_loader)
    val_accuracy = val_correct_predictions / val_total_samples
    
    print('epoch',epoch)
    


    with open(output_filename, 'a') as f:
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}\n')
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n')
        f.write(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}\n')
        f.write(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n')



    # Save model checkpoint every 10 epochs
    if (epoch + 1) % 1== 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, f'models/MCC_{model_name}_{n_filter}/MCC_{model_name}_epoch{epoch+1}.pth')

with open(output_filename, 'a') as f:
    print(f"Training complete. Model saved to 'models/MCC_{model_name}_{n_filter}/MCC_{model_name}_Last.pth\n")
    f.write(f"Training complete. Model saved to 'models/MCC_{model_name}_{n_filter}/MCC_{model_name}_Last.pth\n")

