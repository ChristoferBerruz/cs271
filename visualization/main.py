import click
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


@click.group()
def cli():
    """Command-line interface for visualization."""
    pass


@cli.command('visualize-all')
@click.option('--results-dir', type=click.Path(exists=True, file_okay=False), required=True,
              help='Directory to save visualizations.')
@click.option('--json-paths', type=click.Path(exists=True, dir_okay=False), required=True, multiple=True,
              help='Path to JSON result files.')
@click.option('--epochs', type=int, default=5, help='Epoch number for confusion matrices.')
def visualize_all(results_dir, json_paths, epochs):
    """
    Visualize metrics for classifiers other than ADABOOST.
    """
    # Data storage for aggregated accuracy chart
    aggregated_accuracy_data = []

    for json_path in json_paths:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Access 'experiment_result'
            experiment_result = data.get('experiment_result')
            if experiment_result is None:
                click.echo(f"Failed to generate plots for {json_path}: 'experiment_result' is missing.")
                continue

            # Extract metrics
            training_accuracies = experiment_result.get('training_accuracies')
            training_losses = experiment_result.get('training_losses')
            testing_accuracies = experiment_result.get('testing_accuracies')
            testing_losses = experiment_result.get('testing_losses')
            training_classification_results = experiment_result.get('training_classification_results')
            testing_classification_results = experiment_result.get('testing_classification_results')
            f1_score = experiment_result.get('f1_score')  # Optional

            # Validate presence of required metrics
            required_metrics = {
                'training_accuracies': training_accuracies,
                'training_losses': training_losses,
                'testing_accuracies': testing_accuracies,
                'testing_losses': testing_losses,
                'training_classification_results': training_classification_results,
                'testing_classification_results': testing_classification_results
            }

            missing_metrics = [metric for metric, value in required_metrics.items() if value is None]
            if missing_metrics:
                click.echo(f"Failed to generate plots for {json_path}: Missing metrics: {', '.join(missing_metrics)}.")
                continue

            # Extract classifier and embedding information from 'embedder_name'
            embedder_name = data.get('embedder_name', 'unknown_unknown')
            try:
                embedding_type, class_type = embedder_name.split('_', 1)
            except ValueError:
                embedding_type = embedder_name
                class_type = 'unknown'

            # Append to aggregated data (final epoch's testing accuracy)
            if isinstance(testing_accuracies, list):
                final_testing_accuracy = testing_accuracies[-1]
            else:
                final_testing_accuracy = testing_accuracies
            aggregated_accuracy_data.append({
                'Embedding Type': embedding_type.lower(),
                'Class Type': class_type.lower(),
                'Accuracy': final_testing_accuracy
            })

            # Define plot directory
            plot_dir = os.path.join(results_dir, 'visualizations', 'SIMPLEMLP', class_type.lower(),
                                    embedding_type.lower())
            os.makedirs(plot_dir, exist_ok=True)

            # Plot Training Accuracy
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(training_accuracies) + 1), training_accuracies, marker='o', label='Training Accuracy',
                     color=plt.get_cmap('viridis')(0.2))
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'Training Accuracy over Epochs for SIMPLEMLP ({class_type}, {embedding_type})')
            plt.legend()
            training_accuracy_path = os.path.join(plot_dir, 'training_accuracy.png')
            plt.savefig(training_accuracy_path)
            plt.close()

            # Plot Testing Accuracy
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(testing_accuracies) + 1), testing_accuracies, marker='o', label='Testing Accuracy',
                     color=plt.get_cmap('viridis')(0.6))
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'Testing Accuracy over Epochs for SIMPLEMLP ({class_type}, {embedding_type})')
            plt.legend()
            testing_accuracy_path = os.path.join(plot_dir, 'testing_accuracy.png')
            plt.savefig(testing_accuracy_path)
            plt.close()

            # Plot Training Loss
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(training_losses) + 1), training_losses, marker='o', label='Training Loss',
                     color=plt.get_cmap('viridis')(0.4))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss over Epochs for SIMPLEMLP ({class_type}, {embedding_type})')
            plt.legend()
            training_loss_path = os.path.join(plot_dir, 'training_loss.png')
            plt.savefig(training_loss_path)
            plt.close()

            # Plot Testing Loss
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(testing_losses) + 1), testing_losses, marker='o', label='Testing Loss',
                     color=plt.get_cmap('viridis')(0.8))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Testing Loss over Epochs for SIMPLEMLP ({class_type}, {embedding_type})')
            plt.legend()
            testing_loss_path = os.path.join(plot_dir, 'testing_loss.png')
            plt.savefig(testing_loss_path)
            plt.close()

            # Plot Confusion Matrix for Training (Specific Epoch)
            if isinstance(training_classification_results, list) and len(training_classification_results) >= epochs:
                selected_train_cm = training_classification_results[epochs - 1]
            else:
                selected_train_cm = training_classification_results[-1] if isinstance(training_classification_results,
                                                                                      list) else training_classification_results

            cm_train = []
            labels = sorted(selected_train_cm.keys())
            for label in labels:
                row = []
                for pred_label in labels:
                    row.append(selected_train_cm[label].get(pred_label, 0))
                cm_train.append(row)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (Training) for SIMPLEMLP ({class_type}, {embedding_type}) at Epoch {epochs}')
            cm_train_path = os.path.join(plot_dir, f'confusion_matrix_training_epoch_{epochs}.png')
            plt.savefig(cm_train_path)
            plt.close()

            # Plot Confusion Matrix for Testing (Specific Epoch)
            if isinstance(testing_classification_results, list) and len(testing_classification_results) >= epochs:
                selected_test_cm = testing_classification_results[epochs - 1]
            else:
                selected_test_cm = testing_classification_results[-1] if isinstance(testing_classification_results,
                                                                                    list) else testing_classification_results

            cm_test = []
            labels = sorted(selected_test_cm.keys())
            for label in labels:
                row = []
                for pred_label in labels:
                    row.append(selected_test_cm[label].get(pred_label, 0))
                cm_test.append(row)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (Testing) for SIMPLEMLP ({class_type}, {embedding_type}) at Epoch {epochs}')
            cm_test_path = os.path.join(plot_dir, f'confusion_matrix_testing_epoch_{epochs}.png')
            plt.savefig(cm_test_path)
            plt.close()

            # Plot F1-Score if available
            if f1_score and isinstance(f1_score, list) and len(f1_score) >= epochs:
                final_f1_score = f1_score[-1]
                plt.figure(figsize=(8, 6))
                plt.plot(range(1, len(f1_score) + 1), f1_score, marker='o', label='F1-Score',
                         color=plt.get_cmap('viridis')(0.9))
                plt.xlabel('Epoch')
                plt.ylabel('F1-Score')
                plt.title(f'F1-Score over Epochs for SIMPLEMLP ({class_type}, {embedding_type})')
                plt.legend()
                f1_path = os.path.join(plot_dir, 'f1_score.png')
                plt.savefig(f1_path)
                plt.close()
            else:
                click.echo(f"No F1-Score data available for {json_path}. Skipping F1-Score plot.")

            click.echo(f"Generated all plots for SIMPLEMLP ({class_type}, {embedding_type}) model from {json_path}")

        except json.JSONDecodeError:
            click.echo(f"Failed to parse JSON file: {json_path}")
        except Exception as e:
            click.echo(f"Failed to generate plots for {json_path}: {str(e)}")

    # Generate Accuracy by Embedding Type Chart (Aggregated across classes)
    if aggregated_accuracy_data:
        df_accuracy = pd.DataFrame(aggregated_accuracy_data)
        plt.figure(figsize=(12, 10))
        sns.barplot(data=df_accuracy, x='Embedding Type', y='Accuracy', hue='Class Type', palette='viridis')
        plt.title('Final Testing Accuracy by Embedding Type for SIMPLEMLP')
        plt.ylabel('Accuracy')
        plt.xlabel('Embedding Type')
        plt.legend(title='Class Type')
        plt.tight_layout()
        accuracy_chart_path = os.path.join(results_dir, 'visualizations', 'accuracy_by_embedding.png')
        plt.savefig(accuracy_chart_path)
        plt.close()
        click.echo(f"Saved Accuracy by Embedding Type Chart to {accuracy_chart_path}")
    else:
        click.echo("No accuracy data available to plot Accuracy by Embedding Type.")


@cli.command('visualize-adaboost')
@click.option('--results-dir', type=click.Path(exists=True, file_okay=False), required=True,
              help='Directory to save visualizations.')
@click.option('--json-paths', type=click.Path(exists=True, dir_okay=False), required=True, multiple=True,
              help='Path to ADABOOST JSON result files.')
def visualize_adaboost(results_dir, json_paths):
    """
    Visualize metrics specifically for ADABOOST classifiers.
    """
    # Data storage for aggregated accuracy chart
    aggregated_accuracy_data = []

    for json_path in json_paths:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract necessary data directly from top-level keys
            original_ds_name = data.get('original_ds_name', 'unknown')
            embedder_name = data.get('embedder_name', 'unknown_unknown')
            accuracy = data.get('accuracy')
            classification_makeup = data.get('classification_makeup')

            # Validate presence of required metrics
            if accuracy is None or classification_makeup is None:
                click.echo(f"Failed to generate plots for {json_path}: Missing 'accuracy' or 'classification_makeup'.")
                continue

            # Extract embedding type
            try:
                embedding_type, _ = embedder_name.split('_', 1)
            except ValueError:
                embedding_type = embedder_name

            # Append to aggregated data
            aggregated_accuracy_data.append({
                'Embedding Type': embedding_type.lower(),
                'Accuracy': accuracy
            })

            # Define plot directory
            classifier = 'ADABOOST'
            plot_dir = os.path.join(results_dir, 'visualizations', classifier, embedding_type.lower())
            os.makedirs(plot_dir, exist_ok=True)

            # Plot Accuracy (single bar using Matplotlib)
            plt.figure(figsize=(6, 6))
            plt.bar(['Accuracy'], [accuracy], color=plt.get_cmap('viridis')(0.6))  # Select a specific shade
            plt.ylim(0, 1)
            plt.title(f'Accuracy for ADABOOST ({embedding_type.upper()})')
            plt.ylabel('Accuracy')
            plt.xlabel('')
            plt.tight_layout()
            accuracy_path = os.path.join(plot_dir, 'accuracy.png')
            plt.savefig(accuracy_path)
            plt.close()

            # Plot Confusion Matrix
            # Convert classification_makeup to 2D list
            cm = []
            labels = sorted(classification_makeup.keys())
            for label in labels:
                row = []
                for pred_label in labels:
                    row.append(classification_makeup[label].get(pred_label, 0))
                cm.append(row)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for ADABOOST ({embedding_type.upper()})')
            cm_path = os.path.join(plot_dir, f'confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()

            click.echo(f"Generated plots for ADABOOST ({embedding_type.upper()}) model from {json_path}")

        except json.JSONDecodeError:
            click.echo(f"Failed to parse JSON file: {json_path}")
        except Exception as e:
            click.echo(f"Failed to generate plots for {json_path}: {str(e)}")

    # Generate Accuracy by Embedding Type Chart for ADABOOST
    if aggregated_accuracy_data:
        df_accuracy = pd.DataFrame(aggregated_accuracy_data)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=df_accuracy, x='Embedding Type', y='Accuracy', palette='viridis')
        plt.title('Final Accuracy by Embedding Type for ADABOOST')
        plt.ylabel('Accuracy')
        plt.xlabel('Embedding Type')
        plt.legend().remove()  # Remove legend as hue is not used
        plt.tight_layout()
        accuracy_chart_path = os.path.join(results_dir, 'visualizations', 'ADABOOST', 'accuracy_by_embedding.png')
        plt.savefig(accuracy_chart_path)
        plt.close()
        click.echo(f"Saved ADABOOST Accuracy by Embedding Type Chart to {accuracy_chart_path}")
    else:
        click.echo("No accuracy data available to plot ADABOOST Accuracy by Embedding Type.")


if __name__ == '__main__':
    cli()