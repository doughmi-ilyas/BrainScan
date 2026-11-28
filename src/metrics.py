"""Metrics computed from a confusion matrix (rows = true class, columns = predicted)."""

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def accuracy(matrix):
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    return correct / total


def per_class(matrix, classes=CLASSES):
    """Return {class: {"precision": p, "recall": r, "f1": f}}."""
    result = {}
    for i, name in enumerate(classes):
        tp = matrix[i][i]
        predicted = sum(row[i] for row in matrix)
        actual = sum(matrix[i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[name] = {"precision": precision, "recall": recall, "f1": f1}
    return result


def macro_f1(matrix, classes=CLASSES):
    scores = per_class(matrix, classes)
    return sum(v["f1"] for v in scores.values()) / len(scores)
