import numpy as np

def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    """
    x = x - np.amax(x)
    return np.exp(x / t) / np.sum(np.exp(x) / t)