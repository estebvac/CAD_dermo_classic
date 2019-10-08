import numpy as np

def is_right(image):
    """

    Parameters
    ----------
    image: numpy array
        Image to work with

    Returns
    -------
    boolean
        True if the orientation is right, false otherwise

    """

    left = np.mean(np.array_split(np.sum(image, axis=0), 2)[0])
    right = np.mean(np.array_split(np.sum(image, axis=0), 2)[1])

    return right > left
