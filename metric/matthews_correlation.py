from sklearn.metrics import matthews_corrcoef

def compute_matthews_correlation(predictions, references):
    return {
            "matthews_correlation": float(matthews_corrcoef(references, predictions)),
        }