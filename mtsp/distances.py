import numpy as np


def euclidean(vecA: np.ndarray, vecB: np.ndarray) -> float:
    """
    Calculate the euclidean distance between two one-dimensional vectors.

    Parameters:
        vecA (np.ndarray): Vector A.
        vecB (np.ndarray): Vector B.

    Returns: 
        float: Euclidean distance between vector A and B.

    Raises:
        ValueError: Incompatible vector shapes.
    """
    if 1 not in vecA.shape or 1 not in vecB.shape:
        raise ValueError("Vectors must be one-dimensional arrays.\n" + \
                         "Received shapes: {} and {}".format(vecA, vecB))
    else:
        return np.linalg.norm(np.subtract(vecA, vecB))
