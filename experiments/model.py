#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:14:12 2019

@author: hananhindy
"""
import pandas as pd
import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the neural network model
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

# Helper Function
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Function to load data
def load_data(path, mode, label, verbose=True):
    dataset = pd.read_csv(path)
    
    # Drop unnecessary columns based on mode
    if mode == 0:
        columns_to_drop = ['timestamp', 'src_ip', 'dst_ip']
    elif mode == 1:
        columns_to_drop = ['proto', 'ip_src', 'ip_dst']
    else:
        columns_to_drop = ['proto', 'ip_src', 'ip_dst', 'is_attack']
    
    dataset.drop(columns=columns_to_drop, inplace=True)
    
    # Fill missing values with -1
    dataset.fillna(-1, inplace=True)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    dataset['protocol'] = label_encoder.fit_transform(dataset['protocol'])
    
    # Convert dataframe to numpy array
    data = dataset.values
    
    # Split features and labels
    X = data[:, :-1].astype(float)
    y = data[:, -1].astype(int)
    
    if verbose:
        print(dataset.columns)
    
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=2)
    parser.add_argument('--output', default='Classification_Bi')
    parser.add_argument('--verbose', type=str2bool, default=True)

    args = parser.parse_args()
    
    for slice_number in range(10):
        prefix = 'biflow_' if args.mode == 2 else 'uniflow_'
        
        if args.verbose:
            print('Starting Slice #: {}'.format(slice_number))
            print('Start Classification')
        
        folder_name = '{}_{}/'.format(args.output, slice_number)
        
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
            
        X_normal, y_normal = load_data(prefix + 'normal.csv', args.mode, 0, args.verbose)
        X_scan_A, y_scan_A = load_data(prefix + 'scan_A.csv', args.mode, 1, args.verbose)
        X_scan_sU, y_scan_sU = load_data(prefix + 'scan_sU.csv', args.mode, 2, args.verbose)
        X_sparta, y_sparta = load_data(prefix + 'sparta.csv', args.mode, 3, args.verbose)
        X_mqtt_bruteforce, y_mqtt_bruteforce = load_data(prefix + 'mqtt_bruteforce.csv', args.mode, 4, args.verbose)
        
        X = np.concatenate((X_normal, X_scan_A, X_scan_sU, X_sparta, X_mqtt_bruteforce), axis=0)
        y = np.concatenate((y_normal, y_scan_A, y_scan_sU, y_sparta, y_mqtt_bruteforce), axis=0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create DataLoader for training and testing data
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Define the model
        input_size = X_train.shape[1]
        output_size = len(np.unique(y_train))
        model = Net(input_size, output_size)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            print('Epoch %d, Loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
        
        # Evaluate the model
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.tolist())
                all_labels.extend(labels.tolist())
        
        # Classification report
        report = classification_report(all_labels, all_preds)
        print(report)
        
        # Save the model
        torch.save(model.state_dict(), folder_name + 'model.pt')
