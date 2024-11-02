import torch

from typing import Type, List, ClassVar, Dict, Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from abc import ABC, abstractmethod

from mlproject.datasets import HumanChatBotDataset

from dataclasses import dataclass, field


@dataclass
class NeuralNetworkExperimentResult:
    learning_rate: float
    epochs: int
    training_batch_size: int
    optimizer_name: str
    criterion_name: str
    training_accuracies: List[float] = field(default_factory=list)
    training_losses: List[float] = field(default_factory=list)
    testing_accuracies: List[float] = field(default_factory=list)
    testing_losses: List[float] = field(default_factory=list)

    def save(self, path: str):
        with open(path, mode="wb") as f:
            torch.save(self, f)

    @classmethod
    def load(cls, path: str) -> "NeuralNetworkExperimentResult":
        with open(path, mode="rb") as f:
            return torch.load(f)

    @property
    def training_loss(self):
        return self.training_losses[-1]

    @property
    def testing_loss(self):
        return self.testing_losses[-1]

    @property
    def testing_accuracy(self):
        return self.testing_accuracies[-1]

    @property
    def training_accuracy(self):
        return self.training_accuracies[-1]


class NNBaseModel(torch.nn.Module, ABC):
    registry: ClassVar[Dict[str, "NNBaseModel"]] = {}

    def __init__(self, input_dim: int, output_dim: int):
        super(NNBaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __init_subclass__(cls) -> None:
        if cls.__name__ not in NNBaseModel.registry:
            NNBaseModel.registry[cls.__name__] = cls
        return super().__init_subclass__()

    @abstractmethod
    def train(
        self,
        training_dataset: HumanChatBotDataset,
        testing_dataset: HumanChatBotDataset,
        learning_rate: float = 0.001,
        epochs: int = 100,
    ):
        pass

    def compute_accuracy(self, data_loader: DataLoader) -> Tuple[int, int, float]:
        correct = 0
        with torch.no_grad():
            for text_vectors, text_labels in data_loader:
                outputs = self(text_vectors)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == text_labels).sum().item()
        total_items = len(data_loader.dataset)
        accuracy = correct / total_items
        return correct, total_items, accuracy

    def compute_loss(self, data_loader: DataLoader, criterion: torch.nn.Module) -> float:
        loss = 0.0
        with torch.no_grad():
            for text_vectors, text_labels in data_loader:
                outputs = self(text_vectors)
                loss += criterion(outputs, text_labels).item()
        return loss

    def validate_after_epoch(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            criterion: torch.nn.Module,
            exp_result: NeuralNetworkExperimentResult
    ):
        training_losses = exp_result.training_losses
        testing_losses = exp_result.testing_losses
        training_accuracies = exp_result.training_accuracies
        testing_accuracies = exp_result.testing_accuracies
        training_loss = self.compute_loss(train_loader, criterion)
        training_losses.append(training_loss)
        _, _, training_accuracy = self.compute_accuracy(
            train_loader)
        training_accuracies.append(training_accuracy)
        testing_loss = self.compute_loss(test_loader, criterion)
        testing_losses.append(testing_loss)
        _, _, testing_accuracy = self.compute_accuracy(
            test_loader)
        testing_accuracies.append(testing_accuracy)


class LogisticRegression(NNBaseModel):
    """A simple neural network logistic regression model.
    Different than a MLP, this model has no hidden layers.

    NOT TO BE CONFUSED WITH THE SCIKIT-LEARN LOGISTIC REGRESSION MODEL.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__(input_dim, output_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def train(
        self,
        train_dataset: HumanChatBotDataset,
        test_dataset: HumanChatBotDataset,
        epochs: int,
        learning_rate: float
    ) -> NeuralNetworkExperimentResult:
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        training_batch_size = 32
        train_loader = DataLoader(
            train_dataset, batch_size=training_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print(
            f"Proceeding to train {self.__class__.__name__} for {epochs} epochs...")
        exp_result = NeuralNetworkExperimentResult(
            learning_rate=learning_rate,
            training_batch_size=training_batch_size,
            optimizer_name="SGD",
            criterion_name="CrossEntropyLoss",
            epochs=epochs
        )
        for epoch in range(epochs):
            for _, (text_vectors, text_labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(text_vectors)
                loss = criterion(outputs, text_labels)
                loss.backward()
                optimizer.step()
            self.validate_after_epoch(
                train_loader,
                test_loader,
                criterion,
            )
            training_loss = exp_result.training_loss
            training_accuracy = exp_result.training_accuracy
            testing_loss = exp_result.testing_loss
            testing_accuracy = exp_result.testing_accuracy

            print(f"Epoch: {epoch}, Training Loss: {training_loss}, Training Accuracy: {training_accuracy*100:.2f}%, Testing Loss: {testing_loss}, Testing Accuracy: {testing_accuracy*100:.2f}%")
        return exp_result


class SimpleMLP(NNBaseModel):
    """A simple multi-layer perceptron model with only 2
    neurons in the hidden layer.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(SimpleMLP, self).__init__(input_dim, output_dim)
        self.fc1 = torch.nn.Linear(input_dim, 2)
        self.fc2 = torch.nn.Linear(2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def train(
        self,
        train_dataset: HumanChatBotDataset,
        test_dataset: HumanChatBotDataset,
        epochs: int,
        learning_rate: float
    ):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print(
            f"Proceeding to train {self.__class__.__name__} for {epochs} epochs...")
        exp_result = NeuralNetworkExperimentResult(
            learning_rate=learning_rate,
            optimizer_name="SGD",
            criterion_name="CrossEntropyLoss",
            epochs=epochs,
            training_batch_size=32
        )
        for epoch in range(epochs):
            for _, (text_vectors, text_labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(text_vectors)
                loss = criterion(outputs, text_labels)
                loss.backward()
                optimizer.step()

            self.validate_after_epoch(
                train_loader,
                test_loader,
                criterion,
                exp_result
            )
            training_loss = exp_result.training_loss
            training_accuracy = exp_result.training_accuracy
            testing_loss = exp_result.testing_loss
            testing_accuracy = exp_result.testing_accuracy
            print(f"Epoch: {epoch}, Training Loss: {training_loss}, Training Accuracy: {training_accuracy*100:.2f}%, Testing Loss: {testing_loss}, Testing Accuracy: {testing_accuracy*100:.2f}%")


class CNN2D(NNBaseModel):

    def __init__(self, input_dim: int, output_dim: int):
        # TODO: Conside expanding the neural network with batchnorm
        # dropout, and the like.
        super(CNN2D, self).__init__(input_dim, output_dim)
        self.vector_size = input_dim
        # assuming a square image where input dim (a vector) is reshaped into an image
        self.image_width = int(self.vector_size**0.5)
        self.in_features = 1
        self.conv = torch.nn.Conv2d(
            self.in_features, 2, kernel_size=2, padding=1, stride=1)
        self.fc = torch.nn.Linear(242, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))
        return x

    def train(
        self,
        training_dataset: HumanChatBotDataset,
        testing_dataset: HumanChatBotDataset,
        learning_rate: float = 0.001,
        epochs: int = 100,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        training_batch_size = 32
        train_loader = DataLoader(
            training_dataset, batch_size=training_batch_size, shuffle=True)
        test_loader = DataLoader(
            testing_dataset, batch_size=32, shuffle=False
        )
        exp_result = NeuralNetworkExperimentResult(
            learning_rate=learning_rate,
            training_batch_size=training_batch_size,
            criterion_name="CrossEntropyLoss",
            optimizer_name="Adam",
            epochs=epochs
        )
        for epoch in range(epochs):
            for text_vectors, text_labels in train_loader:
                optimizer.zero_grad()
                outputs = self(text_vectors)
                loss = criterion(outputs, text_labels)
                loss.backward()
                optimizer.step()
            self.validate_after_epoch(
                train_loader,
                test_loader,
                criterion,
                exp_result
            )
            training_loss = exp_result.training_loss
            training_accuracy = exp_result.training_accuracy
            testing_loss = exp_result.testing_loss
            testing_accuracy = exp_result.testing_accuracy
            print(f"Epoch: {epoch}, Training Loss: {training_loss}, Training Accuracy: {training_accuracy*100:.2f}%, Testing Loss: {testing_loss}, Testing Accuracy: {testing_accuracy*100:.2f}%")
