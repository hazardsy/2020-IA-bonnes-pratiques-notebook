import pytest
from source.DataTransformation import one_hot_encode, get_data_and_names, scale, split
import numpy as np


def test_one_hot_encode():
    source_data = np.array(["foo", "bar", "foo", "baz"])
    expected_data = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    actual_data = one_hot_encode(source_data)

    print(expected_data)
    print(actual_data)

    np.testing.assert_array_equal(actual_data, expected_data)


def test_scale():
    source_data = np.array([[1, 2, 3], [4, 5, 750]])
    scaled_data = scale(source_data)

    assert np.mean(scaled_data) == 0
    assert np.var(scaled_data) == 1


def test_split():
    source_X = [1, 2, 3, 4]
    source_Y = [5, 6, 7, 8]
    split_size = 0.25

    (X_train, Y_train), (X_test, Y_test) = split(source_X, source_Y, split_size)

    assert len(X_train) == 3
    assert len(X_test) == 1
    assert len(Y_train) == 3
    assert len(Y_test) == 1


def test_get_data_and_names():
    source_data = [
        {"feat_1": 1, "feat_2": 2, "class": "foo"},
        {"feat_1": 3, "feat_2": 4, "class": "bar"},
    ]

    X, Y, feature_names, target_names = get_data_and_names(source_data)

    expected_Y = [1, 0]
    np.testing.assert_array_equal(X, [[1, 2], [3, 4]])
    assert len(Y) == len(expected_Y) and sorted(Y) == sorted(expected_Y)
    np.testing.assert_array_equal(feature_names, ["feat_1", "feat_2"])
    np.testing.assert_array_equal(target_names, ["foo", "bar"])
