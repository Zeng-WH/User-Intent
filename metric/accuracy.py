import datasets
from sklearn.metrics import accuracy_score


def compute_accuracy(predictions, references):
    return {
        "accuracy": float(
            accuracy_score(references, predictions)
        )
    }

