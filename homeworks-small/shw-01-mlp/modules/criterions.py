import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return np.mean((input - target) ** 2)
        #return super().compute_output(input, target)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return 2 * (input - target) / input.size
        #return super().compute_grad_input(input, target)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        B, C = input.shape
        log_p = self.log_softmax(input)
        true_log_p = log_p[np.arange(B), target]
        return - ( 1 - self.label_smoothing) * np.mean(true_log_p) - (self.label_smoothing / C) * np.mean(np.sum(log_p, axis=1))
        #return super().compute_output(input, target)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        B, C = input.shape
        p = np.exp(self.log_softmax(input))
        grad = p.copy()
        grad[np.arange(B), target] -= (1 - self.label_smoothing)
        grad -= self.label_smoothing / C
        return grad / B
        # return super().compute_grad_input(input, target)
