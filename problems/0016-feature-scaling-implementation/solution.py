import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):

    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0, ddof=0)
    standardized_data = (data - mu) / sigma

    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    normalized_data = (data - min) / (max - min)

    return standardized_data, normalized_data