from typing import Any, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def one_hot_encode(data: np.array) -> np.array:
    """One hot encodes a given input

    Args:
        data (np.array): Original data

    Returns:
        np.array: One hot encoded data
    """
    enc = OneHotEncoder()
    return enc.fit_transform(data[:, np.newaxis]).toarray()


def get_data_and_names(
    data: List[Any], target_col: str = "class"
) -> Tuple[np.array, np.array, List[str], List[str]]:
    """Get X, Y, feature names and target names from a given dataset

    Args:
        data (List[Any]): Original data
        target_col (str): Column containing the target classification

    Returns:
        Tuple[np.array, np.array, List[str], List[str]]: Tuple containing X, Y, feature names and target names
    """
    feature_names = [col for col in data[0].keys() if col not in target_col]
    target_names = list(set(row[target_col] for row in data))

    X = np.array([[float(row[feature]) for feature in feature_names] for row in data])
    y = np.array([target_names.index(row[target_col]) for row in data])

    return X, y, feature_names, target_names


def scale(data: np.array) -> np.array:
    """Scales a given numpy array to have mean 0 and variance 1

    Args:
        data (np.array): Original data

    Returns:
        np.array: Scaled data
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def split(
    X: np.array, Y: np.array, test_size: float = 0.5
) -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    """Performs train test split on given data

    Args:
        X (np.array): Original X input
        Y (np.array): Original Y input
        test_size (float, optional): Size of testing data. Defaults to 0.5.

    Returns:
        Tuple[np.array, np.array]: Tuple of shape ((X_train, Y_train), (X_test, Y_test)) 
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=2
    )
    return (X_train, Y_train), (X_test, Y_test)
