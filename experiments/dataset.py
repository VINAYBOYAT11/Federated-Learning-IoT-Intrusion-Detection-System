import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.drop(columns=['is_attack']).values
        self.y = self.data['is_attack'].values.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]
        if self.transform:
            features = self.transform(features)
        return features, label

def get_custom_dataset(csv_file, transform=None):
    dataset = CustomDataset(csv_file, transform=transform)
    return dataset

def prepare_dataset(csv_file, num_partitions, batch_size, val_ratio=0.1):
    dataset = get_custom_dataset(csv_file)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloaders = []
    valloaders = []
    for _ in range(num_partitions):
        train_subset, val_subset = train_test_split(train_dataset, test_size=int(val_ratio*len(train_dataset)),
                                                    random_state=42, shuffle=True)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        trainloaders.append(train_loader)
        valloaders.append(val_loader)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloaders, valloaders, test_loader

# Usage example:
trainloaders, valloaders, test_loader = prepare_dataset('uniflow_mqtt_bruteforce.csv', num_partitions=5, batch_size=64)
