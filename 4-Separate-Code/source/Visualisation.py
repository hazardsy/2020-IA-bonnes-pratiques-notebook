import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from tensorflow import keras

warnings.simplefilter(action="ignore", category=FutureWarning)
sns.set_style("darkgrid")


def get_training_data_visualisation(
    X: np.array, y: np.array, class_names: List[str], feature_names: List[str]
):
    """Make all visualisation for training data

    Args:
        X (np.array): X data
        y (np.array): y data
        class_names (List[str]): Class names
        feature_names (List[str]): Feature names
    """

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    _get_scatter_plot(X, y, class_names, feature_names, (0, 1))

    plt.subplot(1, 2, 2)
    _get_scatter_plot(X, y, class_names, feature_names, (2, 3))


def _get_scatter_plot(
    X: np.array,
    Y: np.array,
    class_names: List[str],
    feature_names: List[str],
    feature_columns: Tuple[int],
):
    """Make a scatter plot for the given data

    Args:
        X (np.array): X data
        Y (np.array): Y data
        class_names (List[str]): Class names
        feature_names (List[str]): Feature names
        feature_columns (Tuple[int]): Feature numbers to display
    """
    x_col, y_col = feature_columns
    for target, target_name in enumerate(class_names):
        X_plot = X[Y == target]
        sns.scatterplot(
            X_plot[:, x_col], X_plot[:, y_col], marker="o", label=target_name
        )
    plt.xlabel(feature_names[x_col])
    plt.ylabel(feature_names[y_col])
    plt.axis("equal")
    plt.legend()


def get_training_plots(data: Dict):
    """Plots validation accuracy and loss

    Args:
        data (Dict): Training metrics
    """
    _, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

    for model_name, history in data.items():
        val_accuracy = history.get("val_accuracy")
        val_loss = history.get("val_loss")
        ax1.plot(val_accuracy, label=model_name)
        ax2.plot(val_loss, label=model_name)

    ax1.set_ylabel("validation accuracy")
    ax2.set_ylabel("validation loss")
    ax2.set_xlabel("epochs")
    ax1.legend()
    ax2.legend()


def get_roc_auc_curve(model_names: List[str], test_data: Tuple[np.array, np.array]):
    """Generates and plots the ROC AUC curve

    Args:
        model_names (List[str]): Name of models to plot
        test_data (Tuple[np.array, np.array]): X_test and Y_test
    """
    X_test, Y_test = test_data
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k--")

    for model_name in model_names:
        model = keras.models.load_model(f"../models/{model_name}")

        Y_pred = model.predict(X_test)
        fpr, tpr, _ = roc_curve(Y_test.ravel(), Y_pred.ravel())

        plt.plot(fpr, tpr, label="{}, AUC = {:.3f}".format(model_name, auc(fpr, tpr)))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
