import torch

from typing import Type, List
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class LogisticRegression(torch.nn.Module):
    """A simple neural network logistic regression model.
    Different than a MLP, this model has no hidden layers.

    NOT TO BE CONFUSED WITH THE SCIKIT-LEARN LOGISTIC REGRESSION MODEL.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer_klass: Type[Optimizer],
        criterion_klass: Type[torch.nn.Module],
        epochs: int,
        learning_rate: float
    ):
        losses: List[float] = []
        accuracies: List[float] = []
        optimizer = optimizer_klass(self.parameters(), lr=learning_rate)
        criterion = criterion_klass()
        print(
            f"Proceeding to train {self.__class__.__name__} for {epochs} epochs...")
        for epoch in range(epochs):
            for _, (text_vectors, text_labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(text_vectors)
                loss = criterion(outputs, text_labels)
                loss.backward()
                optimizer.step()
            # get the loss after a batch of data is processed
            losses.append(loss.item())
            # compute current accuracy
            correct = 0
            for text_vectors, text_labels in test_loader:
                outputs = self(text_vectors)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == text_labels).sum().item()
            accuracy = correct / len(test_loader.dataset)
            accuracies.append(accuracy)
            print(
                f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy*100:.2f}%")


class SimpleMLP(torch.nn.Module):
    """A simple multi-layer perceptron model with only 2
    neurons in the hidden layer.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(SimpleMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 2)
        self.fc2 = torch.nn.Linear(2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer_klass: Type[Optimizer],
        criterion_klass: Type[torch.nn.Module],
        epochs: int,
        learning_rate: float
    ):
        losses: List[float] = []
        accuracies: List[float] = []
        optimizer = optimizer_klass(self.parameters(), lr=learning_rate)
        criterion = criterion_klass()
        print(
            f"Proceeding to train {self.__class__.__name__} for {epochs} epochs...")
        for epoch in range(epochs):
            for _, (text_vectors, text_labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(text_vectors)
                loss = criterion(outputs, text_labels)
                loss.backward()
                optimizer.step()
            # get the loss after a batch of data is processed
            losses.append(loss.item())
            # compute current accuracy
            correct = 0
            for text_vectors, text_labels in test_loader:
                outputs = self(text_vectors)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == text_labels).sum().item()
            accuracy = correct / len(test_loader.dataset)
            accuracies.append(accuracy)
            print(
                f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy*100:.2f}%")
