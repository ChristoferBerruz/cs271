# data_viz.py

"""
This module contains all classes and functions related to producing
visualizations for model training data.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import scienceplots

plt.style.use(['science', 'ieee'])
plt.rcParams.update({
    'text.usetex': False
})


class DataViz(ABC):
    """
    An abstract base class to visualize the results of model training data.
    """

    @abstractmethod
    def visualize(self, experiment: Dict[str, Any], image_path: str, **kwargs) -> None:
        """
        Visualize the results of model training data.

        Args:
            experiment (Dict[str, Any]): The experiment result data.
            image_path (str): The path to save the image.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        pass

    @classmethod
    def load_json(cls, json_path: str) -> Dict[str, Any]:
        """
        Load training data from a JSON file.

        Args:
            json_path (str): The path to the training data .json file.

        Returns:
            Dict[str, Any]: The loaded JSON data as a dictionary.
        """
        print(f"Loading training data from {json_path}")
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        return data


class AccuracyPlot(DataViz):
    """
    A class to visualize training and testing accuracies over epochs.
    """

    def visualize(self, experiment: Dict[str, Any], image_path: str, **kwargs) -> None:
        """
        Plot training and testing accuracies over epochs.

        Args:
            experiment (Dict[str, Any]): The experiment result data.
            image_path (str): The path to save the accuracy plot.
            **kwargs: Additional keyword arguments (e.g., metadata for title).

        Returns:
            None
        """
        epochs = list(range(1, experiment['epochs'] + 1))
        training_accuracies = experiment['training_accuracies']
        testing_accuracies = experiment.get('testing_accuracies', [])

        plt.figure(figsize=(3.5, 2.5))  # Single-column width
        plt.plot(epochs, training_accuracies, label='Training Accuracy',
                 marker='o', color='blue')
        if testing_accuracies:
            plt.plot(epochs, testing_accuracies, label='Testing Accuracy',
                     marker='s', color='red')
        # Incorporate metadata into the title if provided
        title_suffix = kwargs.get('title_suffix', '')
        plt.title(f'Training and Testing Accuracy over Epochs {title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(image_path, format='png')
        plt.close()
        print(f"Saved Accuracy Plot to {image_path}")


class LossPlot(DataViz):
    """
    A class to visualize training and testing losses over epochs.
    """

    def visualize(self, experiment: Dict[str, Any], image_path: str, **kwargs) -> None:
        """
        Plot training and testing losses over epochs.

        Args:
            experiment (Dict[str, Any]): The experiment result data.
            image_path (str): The path to save the loss plot.
            **kwargs: Additional keyword arguments (e.g., metadata for title).

        Returns:
            None
        """

        epochs = list(range(1, experiment['epochs'] + 1))
        training_losses = experiment['training_losses']
        testing_losses = experiment.get('testing_losses', [])

        plt.figure(figsize=(3.5, 2.5))  # Single-column width
        plt.plot(epochs, training_losses, label='Training Loss',
                 marker='o', color='blue')
        if testing_losses:
            plt.plot(epochs, testing_losses, label='Testing Loss',
                     marker='s', color='red')
        # Incorporate metadata into the title if provided
        title_suffix = kwargs.get('title_suffix', '')
        plt.title(f'Training and Testing Loss over Epochs {title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(image_path, format='png')
        plt.close()
        print(f"Saved Loss Plot to {image_path}")


class ConfusionMatrixPlot(DataViz):
    """
    A class to visualize confusion matrices for a specified epoch.
    """

    def __init__(self, epoch: Optional[int] = None):
        """
        Initialize the ConfusionMatrixPlot.

        Args:
            epoch (Optional[int]): The epoch number to visualize.
                                    If None, defaults to the last epoch.
        """
        self.epoch = epoch

    def visualize(self, experiment: Dict[str, Any], image_prefix: str, class_labels: List[str]) -> None:
        """
        Plot training and testing confusion matrices for a specified epoch.

        Args:
            experiment (Dict[str, Any]): The experiment result data.
            image_prefix (str): The prefix for saving the confusion matrix plots.
            class_labels (List[str]): List of class names for labeling.

        Returns:
            None
        """

        total_epochs = experiment['epochs']
        epoch_to_inspect = self.epoch if self.epoch else total_epochs
        if epoch_to_inspect < 1 or epoch_to_inspect > total_epochs:
            raise ValueError(
                f"Epoch {epoch_to_inspect} is out of range. Must be between 1 and {total_epochs}.")

        # Extract confusion matrices
        train_conf_matrix = experiment.get('training_classification_results', [])
        test_conf_matrix = experiment.get('testing_classification_results', [])

        if not train_conf_matrix or not test_conf_matrix:
            raise ValueError("Confusion matrices not found in the experiment data.")

        if epoch_to_inspect > len(train_conf_matrix):
            raise ValueError(
                f"Epoch {epoch_to_inspect} exceeds the number of available confusion matrices.")

        train_cm = train_conf_matrix[epoch_to_inspect - 1]
        test_cm = test_conf_matrix[epoch_to_inspect - 1]

        # Convert to DataFrame
        train_df = self.dict_to_df(train_cm)
        test_df = self.dict_to_df(test_cm)

        # Plot Training Confusion Matrix
        train_save_path = f"{image_prefix}_training_epoch_{epoch_to_inspect}.png"
        self.plot_confusion_matrix(train_df,
                                   f'Training Confusion Matrix at Epoch {epoch_to_inspect}',
                                   train_save_path,
                                   class_labels)

        # Plot Testing Confusion Matrix
        test_save_path = f"{image_prefix}_testing_epoch_{epoch_to_inspect}.png"
        self.plot_confusion_matrix(test_df,
                                   f'Testing Confusion Matrix at Epoch {epoch_to_inspect}',
                                   test_save_path,
                                   class_labels)

    @staticmethod
    def dict_to_df(conf_dict: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """
        Convert nested confusion matrix dictionary to a DataFrame.

        Args:
            conf_dict (Dict[str, Dict[str, int]]): Nested confusion matrix.

        Returns:
            pd.DataFrame: Confusion matrix as a DataFrame.
        """
        classes = sorted(conf_dict.keys(), key=lambda x: int(x))
        data = {cls: conf_dict[cls] for cls in classes}
        df = pd.DataFrame(data).fillna(0).astype(int)
        return df

    @staticmethod
    def plot_confusion_matrix(df: pd.DataFrame, title: str, save_path: str, class_labels: List[str]) -> None:
        """
        Plot and save a confusion matrix.

        Args:
            df (pd.DataFrame): Confusion matrix as a DataFrame.
            title (str): Title of the plot.
            save_path (str): Path to save the plot.
            class_labels (List[str]): List of class names for labeling.

        Returns:
            None
        """

        plt.figure(figsize=(3.0, 3.0))  # Smaller size for clarity
        sns.heatmap(df, annot=True, fmt='d', cmap='Blues', cbar=False,
                    annot_kws={"size": 6}, xticklabels=class_labels, yticklabels=class_labels)
        plt.title(title, fontsize=10)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.savefig(save_path, format='png')
        plt.close()
        print(f"Saved Confusion Matrix Plot to {save_path}")


class F1ScorePlot(DataViz):
    """
    A class to visualize training and testing F1-Scores over epochs.
    """

    def visualize(self, experiment: Dict[str, Any], image_path: str, **kwargs) -> None:
        """
        Calculate and plot training and testing F1-Scores over epochs.

        Args:
            experiment (Dict[str, Any]): The experiment result data.
            image_path (str): The path to save the F1-Score plot.
            **kwargs: Additional keyword arguments (e.g., metadata for title).

        Returns:
            None
        """

        epochs = list(range(1, experiment['epochs'] + 1))
        f1_scores_train = []
        f1_scores_test = []

        training_confusion = experiment.get('training_classification_results', [])
        testing_confusion = experiment.get('testing_classification_results', [])

        for epoch in range(experiment['epochs']):
            # Training Confusion Matrix
            if epoch >= len(training_confusion):
                print(
                    f"Epoch {epoch + 1} exceeds available training confusion matrices.")
                break
            train_cm = training_confusion[epoch]
            y_true_train, y_pred_train = self.extract_labels(train_cm)
            f1_train = f1_score(y_true_train, y_pred_train, average='weighted')
            f1_scores_train.append(f1_train)

            # Testing Confusion Matrix
            if epoch >= len(testing_confusion):
                print(
                    f"Epoch {epoch + 1} exceeds available testing confusion matrices.")
                break
            test_cm = testing_confusion[epoch]
            y_true_test, y_pred_test = self.extract_labels(test_cm)
            f1_test = f1_score(y_true_test, y_pred_test, average='weighted')
            f1_scores_test.append(f1_test)

        # Plot F1-Scores
        plt.figure(figsize=(3.5, 2.5))  # Single-column width
        plt.plot(epochs[:len(f1_scores_train)], f1_scores_train, label='Training F1-Score',
                 marker='o', color='green')
        plt.plot(epochs[:len(f1_scores_test)], f1_scores_test, label='Testing F1-Score',
                 marker='s', color='orange')
        # Incorporate metadata into the title if provided
        title_suffix = kwargs.get('title_suffix', '')
        plt.title(f'Training and Testing F1-Score over Epochs {title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('F1-Score')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(image_path, format='png')
        plt.close()
        print(f"Saved F1-Score Plot to {image_path}")

    @staticmethod
    def extract_labels(conf_matrix: Dict[str, Dict[str, int]]) -> Tuple[List[int], List[int]]:
        """
        Extract true and predicted labels from a confusion matrix.

        Args:
            conf_matrix (Dict[str, Dict[str, int]]): Nested confusion matrix.

        Returns:
            Tuple[List[int], List[int]]: Lists of true and predicted labels.
        """
        y_true = []
        y_pred = []
        for true_class, preds in conf_matrix.items():
            for pred_class, count in preds.items():
                y_true.extend([int(true_class)] * count)
                y_pred.extend([int(pred_class)] * count)
        return y_true, y_pred


class AdaBoostPlot(DataViz):
    """
    A class to visualize AdaBoost results.
    """

    def visualize(self, experiment: Dict[str, Any], image_prefix: str, class_labels: Optional[List[str]] = None) -> None:
        """
        Plot AdaBoost accuracy and confusion matrix.

        Args:
            experiment (Dict[str, Any]): The experiment result data.
            image_prefix (str): The base path to save the AdaBoost visualizations.
            class_labels (Optional[List[str]]): List of class names for labeling.

        Returns:
            None
        """

        # Plot Accuracy
        accuracy = experiment.get('accuracy', None)
        if accuracy is not None:
            plt.figure(figsize=(3.5, 2.5))
            plt.bar(['AdaBoost Accuracy'], [accuracy], color='purple')
            plt.ylim(0, 1)
            plt.ylabel('Accuracy')
            plt.title('AdaBoost Model Accuracy')
            plt.tight_layout()
            accuracy_save_path = f"{image_prefix}_accuracy.png"
            plt.savefig(accuracy_save_path, format='png')
            plt.close()
            print(f"Saved AdaBoost Accuracy Plot to {accuracy_save_path}")
        else:
            print("Accuracy not found in the AdaBoost experiment data.")

        # Plot Confusion Matrix
        classification_makeup = experiment.get('classification_makeup', {})
        if classification_makeup:
            cm_df = self.dict_to_df(classification_makeup)
            confusion_save_path = f"{image_prefix}_confusion_matrix.png"
            self.plot_confusion_matrix(cm_df, 'AdaBoost Confusion Matrix', confusion_save_path, class_labels)
        else:
            print("Classification makeup not found in the AdaBoost experiment data.")

    @staticmethod
    def dict_to_df(conf_dict: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """
        Convert nested confusion matrix dictionary to a DataFrame.

        Args:
            conf_dict (Dict[str, Dict[str, int]]): Nested confusion matrix.

        Returns:
            pd.DataFrame: Confusion matrix as a DataFrame.
        """
        classes = sorted(conf_dict.keys(), key=lambda x: int(x))
        data = {cls: conf_dict[cls] for cls in classes}
        df = pd.DataFrame(data).fillna(0).astype(int)
        return df

    @staticmethod
    def plot_confusion_matrix(df: pd.DataFrame, title: str, save_path: str, class_labels: Optional[List[str]] = None) -> None:
        """
        Plot and save a confusion matrix.

        Args:
            df (pd.DataFrame): Confusion matrix as a DataFrame.
            title (str): Title of the plot.
            save_path (str): Path to save the plot.
            class_labels (Optional[List[str]]): List of class names for labeling.

        Returns:
            None
        """

        plt.figure(figsize=(3.0, 3.0))  # Smaller size for clarity
        sns.heatmap(df, annot=True, fmt='d', cmap='Greens', cbar=False,
                    annot_kws={"size": 6}, xticklabels=class_labels, yticklabels=class_labels)
        plt.title(title, fontsize=10)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.savefig(save_path, format='png')
        plt.close()
        print(f"Saved AdaBoost Confusion Matrix Plot to {save_path}")