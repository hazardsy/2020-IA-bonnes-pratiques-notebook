from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_model(
    input_dim: int, output_dim: int, nodes: int, n: int = 1, name: str = "model"
) -> Sequential:
    """Create a keras Sequential model

    Args:
        input_dim (int): Input dimension of the model
        output_dim (int): Output dimension of the model
        nodes (int): Number of nodes
        n (int, optional): Number of layers. Defaults to 1.
        name (str, optional): Model name. Defaults to 'model'.

    Returns:
        Sequential: [description]
    """
    model = Sequential(name=name)
    for _ in range(n):
        model.add(Dense(nodes, input_dim=input_dim, activation="relu"))
    model.add(Dense(output_dim, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def train_model(
    model: Sequential,
    X: Tuple[np.array, np.array],
    Y: Tuple[np.array, np.array],
    callbacks: List[Callable] = [],
) -> Dict:
    X_train, X_test = X
    Y_train, Y_test = Y
    """Trains and saves a model

    Args:
        model (Sequential): Model to train
        X (Tuple[np.array, np.array]): X_train and X_test
        Y (Tuple[np.array, np.array]): Y_train and Y_test
        callbacks (List[Callable]): Training callbacks

    Returns:
        Dict: Training metrics
    """

    print("Model name:", model.name)
    history_callback = model.fit(
        X_train,
        Y_train,
        batch_size=5,
        epochs=50,
        verbose=0,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
    )
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print()

    model.save(f"../models/{model.name}")
    return history_callback.history
