import pytest

from src.metrics import accuracy, macro_f1, per_class

# Test-set confusion matrix of the README (assets/confusion_matrix.png), 1,054 images.
MATRIX = [
    [233, 8, 0, 2],
    [20, 212, 7, 8],
    [0, 6, 293, 1],
    [0, 3, 0, 261],
]


def test_accuracy_matches_readme():
    assert accuracy(MATRIX) == pytest.approx(999 / 1054)
    assert round(accuracy(MATRIX) * 100, 1) == 94.8


def test_per_class_matches_readme():
    scores = per_class(MATRIX)
    assert round(scores["meningioma"]["recall"] * 100, 1) == 85.8
    assert round(scores["glioma"]["precision"] * 100, 1) == 92.1
    assert round(scores["pituitary"]["precision"] * 100, 1) == 96.0
    assert round(scores["notumor"]["recall"] * 100, 1) == 97.7


def test_macro_f1_between_zero_and_one():
    assert 0.9 < macro_f1(MATRIX) < 1.0


def test_perfect_matrix():
    identity = [[5, 0], [0, 7]]
    assert accuracy(identity) == 1.0
    assert macro_f1(identity, ["a", "b"]) == 1.0
