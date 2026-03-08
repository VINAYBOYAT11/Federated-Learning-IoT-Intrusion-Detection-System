# client.py

import torch
from torch.utils.data import DataLoader

import flwr as fl
from model import MQTTIDSModel


class MQTTIDSClient(fl.client.NumPyClient):
    def __init__(self, trainloader: DataLoader, valloader: DataLoader) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MQTTIDSModel(num_classes=2).to(self.device)

    def get_parameters(self):
        return [param.cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for (name, value), param in zip(state_dict.items(), parameters):
            state_dict[name] = torch.tensor(param)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Define optimizer and loss function
        optimizer = torch.optim.SGD(self.model.parameters(), lr=config["lr"])
        criterion = torch.nn.CrossEntropyLoss()

        # Train model
        self.model.train()
        for _ in range(config["epochs"]):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Return updated parameters and the number of training examples
        return self.get_parameters(), len(self.trainloader.dataset)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Evaluate model
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.valloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        # Return the accuracy and the number of validation examples
        return float(accuracy), int(total)
