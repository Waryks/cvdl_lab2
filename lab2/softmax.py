from matplotlib.colors import NoNorm
import numpy as np
import matplotlib.pyplot as plt
from activations import softmax

class SoftmaxClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.W = None
        self.initialize()

    def initialize(self):
        # TODO your code here
        # initialize the weight matrix (remember the bias trick) with small random variables
        # you might find np.random.randn userful here *0.001
        self.W = np.random.randn(self.input_shape, self.num_classes) * 0.001

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = None
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X
        # remember about the bias trick!
        # 1. apply the softmax function on the scores
        # 2. returned the normalized scores

        scores = self.score(X)
        scores = softmax(scores)

        return scores

    def score(self, X) -> np.ndarray:
        return np.dot(X, self.W)

    def predict(self, X: np.ndarray) -> int:
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X as the scores
        # 1. compute the prediction by taking the argmax of the class scores

        probs = self.predict_proba(X)

        return np.argmax(probs, axis=1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            **kwargs) -> dict:

        history = []

        bs = kwargs['bs'] if 'bs' in kwargs else 128
        reg_strength = kwargs['reg_strength'] if 'reg_strength' in kwargs else 1e3
        steps = kwargs['steps'] if 'steps' in kwargs else 100
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
        print(bs, reg_strength, steps, lr)

        # run mini-batch gradient descent
        for iteration in range(0, steps):
            # TODO your code here
            # sample a batch of images from the training set
            # you might find np.random.choice useful
            indices = np.random.choice(np.arange(np.shape(X_train)[0]), size=bs, replace=False)
            X_batch, y_batch = X_train[indices], y_train[indices]
            soft = self.predict_proba(X_batch)
            loss = np.sum(-np.log(soft[np.arange(bs), y_batch]))
            loss = loss / bs + reg_strength * np.sum(self.W * self.W)
            y_one_hot = np.zeros((y_batch.size, 10))
            y_one_hot[np.arange(bs), y_batch] = 1
            CT = soft - y_one_hot
            dW = X_batch.T.dot(CT)
            dW = dW / bs + reg_strength * self.W
            # print(loss)
            # end TODO your code here
            # compute the loss and dW
            # perform a parameter update
            self.W -= lr * dW
            # append the training loss, accuracy on the training set and accuracy on the test set to the history dict
            history.append(loss)

        return history


    def get_weights(self, img_shape):
        # TODO your code here
        # 0. ignore the bias term
        # 1. reshape the weights to (*image_shape, num_classes)
        return self.W[:-1, :].reshape(np.append([self.num_classes], img_shape))

    def load(self, path: str) -> bool:
        # TODO your code here
        # load the input shape, the number of classes and the weight matrix from a file
        self.W = np.load(path)
        return True

    def save(self, path: str) -> bool:
        # TODO your code here
        # save the input shape, the number of classes and the weight matrix to a file
        # you might find np.save useful for this
        # TODO your code here
        np.save(path, self.W)

        return True

