"""
Ref: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
Ref: https://towardsdatascience.com/deriving-backpropagation-with-cross-entropy-loss-d24811edeaf9
"""

from typing import Callable, Self, Literal
import numpy as np
import pickle
import wandb
from numpy import floating
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, \
    root_mean_squared_error, r2_score, mean_absolute_error, hamming_loss


class ActivationFunction:
    def __init__(self, activation_function):
        self.forward: Callable
        self.backward: Callable

        if activation_function == "sigmoid":
            self.forward = lambda x: 1 / (1 + np.exp(-x))
            self.backward = lambda z: self.forward(z) * (1 - self.forward(z))
        elif activation_function == "tanh":
            self.forward = lambda x: np.tanh(x)
            self.backward = lambda z: 1 - (np.tanh(z) ** 2)
        elif activation_function == "relu":
            self.forward = lambda x: np.maximum(0, x)
            self.backward = lambda z: (z > 0).astype(float)
        elif activation_function == "linear":
            self.forward = lambda x: x
            self.backward = lambda z: 1


class MLP:
    def __init__(self, objective: Literal['classification', 'multi-label classification', 'regression'],
                 n_neurons_input_layer: int, n_neurons_output_layer: int,
                 n_hidden_layers: int = 1, n_neurons_per_layer: list[int] = [20], learning_rate_init: float = 0.001,
                 activation: Literal['sigmoid', 'tanh', 'relu', 'linear'] = 'relu',
                 optimizer_type: Literal['sgd', 'bgd', 'mbgd'] = 'sgd', batch_size: int = 32,
                 loss_type: Literal['MSE', 'BCE'] = None, max_iter: int = 200, random_state: int = None,
                 tol: float = None, patience: int = 1, enable_logging: bool = False):
        """
        Initialise a Multi-Layer Perceptron object for either classification or regression tasks.

        :param objective: Objective of our MLP. Choose from {'classification', 'multi-label classification', 'regression'}.
        :param n_neurons_input_layer: Number of neurons in the input layer. Should be as many as number of input features.
        :param n_neurons_output_layer: Number of neurons in the output layer. Should be as many as number of output features.
        :param n_hidden_layers: 1 by default. Number of neuron layers in between the input and output layers.
        :param n_neurons_per_layer: [20] by default. List of integers specifying the number of neurons in each hidden layer.
        :param learning_rate_init: 0.001 by default.
        :param activation: 'relu' by default. Choose from {'sigmoid', 'tanh', 'relu', 'linear'}.
        :param optimizer_type: 'sgd' by default. Choose from {'sgd', 'bgd', 'mbgd'}. Here, sgd refers to Stochastic Gradient Descent, bgd refers to Batch Gradient Descent, mbgd refers to Mini Batch Gradient Descent
        :param batch_size: 32 by default.
        :param loss_type: None by default. Only for Regression objective, choose from {'MSE', 'BCE'}.
        :param max_iter: 200 by default.
        :param random_state: None by default. Used to deterministically obtain the weights that are initialised at the start.
        :param tol: None by default. Used for early stopping, if a value is set.
        :param patience: 1 by default. Number of consecutive iterations that the difference between successive loss is less than threshold, before early stopping is triggered.
        :param enable_logging: False by default. Used to enable logging via WandB.
        """
        self.objective: str = objective
        self.n_hidden_layers: int = n_hidden_layers
        self.n_neurons_per_layer: list[int] = n_neurons_per_layer[:n_hidden_layers]  # remove extra vals if any.
        self.learning_rate: float = learning_rate_init
        self.activation_function_chosen: str = activation
        self.activation = ActivationFunction(activation_function=activation)
        self.optimizer_type: str = optimizer_type
        self.batch_size: int = batch_size
        self.loss_type: str = loss_type
        self.max_iter: int = max_iter
        self.random_state: int = random_state
        self.tol: float = tol
        self.enable_logging: bool = enable_logging
        self.patience = patience

        self.n_neurons_input_layer: int = n_neurons_input_layer
        self.n_neurons_output_layer: int = n_neurons_output_layer
        self.weights = []
        self.biases = []
        self._init_weights_and_biases()

    def _init_weights_and_biases(self) -> None:
        self.weights = []
        self.biases = []

        if self.n_hidden_layers > 0:
            # input layer -> first hidden layer
            self.weights.append(
                np.random.RandomState(self.random_state).randn(self.n_neurons_input_layer, self.n_neurons_per_layer[0]))
            self.biases.append(np.zeros((1, self.n_neurons_per_layer[0])))

            # hidden layers
            for i in range(self.n_hidden_layers - 1):
                self.weights.append(np.random.RandomState(self.random_state).randn(self.n_neurons_per_layer[i],
                                                                                   self.n_neurons_per_layer[i + 1]))
                self.biases.append(np.zeros((1, self.n_neurons_per_layer[i + 1])))

            # final hidden layer -> output layer
            self.weights.append(np.random.RandomState(self.random_state).randn(self.n_neurons_per_layer[-1],
                                                                               self.n_neurons_output_layer))
            self.biases.append(np.zeros((1, self.n_neurons_output_layer)))
        else:
            self.weights.append(
                np.random.RandomState(self.random_state).randn(self.n_neurons_input_layer, self.n_neurons_output_layer))
            self.biases.append(np.zeros((1, self.n_neurons_output_layer)))

    def _softmax(self, y) -> float:
        exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
        return exp_y / exp_y.sum(axis=1, keepdims=True)

    def forward_propagation(self, X: np.ndarray) -> tuple[np.ndarray, list, list]:
        next_layer_input = X
        activation_vals = [X]
        z_vals = []

        # in the following loop, I will take synapse_idx = 0 to be input layer -> first hidden layer,
        # and synapse_idx = self.n_hidden_layers to be final hidden layer -> output layer synapse.
        for synapse_idx in range(0, self.n_hidden_layers + 1):
            next_layer_input = next_layer_input @ self.weights[synapse_idx] + self.biases[synapse_idx]
            z_vals.append(next_layer_input)
            next_layer_input = self.activation.forward(next_layer_input)
            activation_vals.append(next_layer_input)

        activation_vals.pop()  # remove y_hat

        if self.objective == "classification" and self.loss_type is None:
            y_hat = self._softmax(z_vals[-1])  # softmax for the output synapses
        elif self.objective == "multi-label classification":
            y_hat = ActivationFunction('sigmoid').forward(z_vals[-1])  # sigmoid for the output synapses
        elif self.objective == "regression" or self.loss_type is not None:
            y_hat = next_layer_input  # same activation function as other synapses

        return y_hat, z_vals, activation_vals  # final layer value, all z_vals and all activation_vals.

    def backward_propagation(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray, z_vals: list,
                             activation_vals: list) -> tuple[list, list]:
        m = X.shape[0]

        if (
                self.objective == "classification" or self.objective == "multi-label classification") and self.loss_type is None:
            delta = (y_hat - y)  # softmax's derivative, but also the case for multi-label.
        elif self.objective == "regression" or self.loss_type is not None:
            # initial delta val, computing separately as formula changes a lil bit later on
            delta = (y_hat - y) * self.activation.backward(z_vals[-1])
        d_weights = []
        d_biases = []

        for layer in reversed(range(self.n_hidden_layers + 1)):
            d_weights.append((activation_vals[layer].T @ delta) / m)
            d_biases.append(np.sum(delta, axis=0, keepdims=True) / m)

            if layer > 0:
                delta = (delta @ self.weights[layer].T) * self.activation.backward(z_vals[layer - 1])

        d_weights.reverse()
        d_biases.reverse()

        return d_weights, d_biases

    def _update_gradients(self, d_weights: list, d_biases: list) -> None:
        for layer in range(self.n_hidden_layers + 1):
            self.weights[layer] -= self.learning_rate * d_weights[layer]
            self.biases[layer] -= self.learning_rate * d_biases[layer]

    def _cost(self, X: np.ndarray, y: np.ndarray) -> float | floating:
        y_hat = self.forward_propagation(X)[0]
        epsilon = 1e-15  # small value for numerical stability
        if self.objective == "classification" and self.loss_type is None:
            # cross-entropy
            return -np.sum(y * np.log(y_hat + epsilon)) / X.shape[0]
        elif self.objective == "multi-label classification" or self.loss_type == "BCE":
            # binary cross entropy
            return -np.mean(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
        elif self.objective == "regression" or self.loss_type == "MSE":
            # MSE
            return 0.5 * np.sum((y - y_hat) ** 2) / X.shape[0]

    def _fit_handler(self, X: np.ndarray, y: np.ndarray) -> None:
        y_hat, z_vals, activation_vals = self.forward_propagation(X)
        d_weights, d_biases = self.backward_propagation(X, y, y_hat, z_vals, activation_vals)
        self._update_gradients(d_weights, d_biases)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray = None,
            y_validation: np.ndarray = None) -> list:
        """
        To initialize the model, and fit it to the training data using the specified gradient descent method.
        :param X_train: Input training data
        :param y_train: Training labels
        """
        current_patience = 1

        costs = []

        if self.enable_logging:
            y_hat_train = self.predict(X_train)
            y_hat_validation = self.predict(X_validation)

            if self.objective == "classification":
                wandb.log(
                    {'epoch': 0, 'train/loss': self._cost(X_train, y_train),
                     'train/accuracy': accuracy_score(y_train, y_hat_train),
                     'train/precision': precision_score(y_train, y_hat_train, average='micro'),
                     'train/recall': recall_score(y_train, y_hat_train, average='micro'),
                     'train/f1_score': f1_score(y_train, y_hat_train, average='micro'),
                     'validation/loss': self._cost(X_validation, y_validation),
                     'validation/accuracy': accuracy_score(y_validation, y_hat_validation),
                     'validation/precision': precision_score(y_validation, y_hat_validation, average='micro'),
                     'validation/recall': recall_score(y_validation, y_hat_validation, average='micro'),
                     'validation/f1_score': f1_score(y_validation, y_hat_validation, average='micro')})
            elif self.objective == "multi-label classification":
                wandb.log(
                    {'epoch': 0, 'train/loss': self._cost(X_train, y_train),
                     'train/accuracy': accuracy_score(y_train, y_hat_train),
                     'train/precision': precision_score(y_train, y_hat_train, average='micro'),
                     'train/recall': recall_score(y_train, y_hat_train, average='micro'),
                     'train/f1_score': f1_score(y_train, y_hat_train, average='micro'),
                     'train/hamming_loss': hamming_loss(y_train, y_hat_train),
                     'validation/loss': self._cost(X_validation, y_validation),
                     'validation/accuracy': accuracy_score(y_validation, y_hat_validation),
                     'validation/precision': precision_score(y_validation, y_hat_validation, average='micro'),
                     'validation/recall': recall_score(y_validation, y_hat_validation, average='micro'),
                     'validation/f1_score': f1_score(y_validation, y_hat_validation, average='micro'),
                     'validation/hamming_loss': hamming_loss(y_validation, y_hat_validation)})
            elif self.objective == "regression":
                wandb.log(
                    {'epoch': 0, 'train/loss': self._cost(X_train, y_train),
                     'train/MSE': mean_squared_error(y_train, y_hat_train),
                     'train/RMSE': root_mean_squared_error(y_train, y_hat_train),
                     'train/R2': r2_score(y_train, y_hat_train),
                     'validation/loss': self._cost(X_validation, y_validation),
                     'validation/MSE': mean_squared_error(y_validation, y_hat_validation),
                     'validation/RMSE': root_mean_squared_error(y_validation, y_hat_validation),
                     'validation/R2': r2_score(y_validation, y_hat_validation)})

        for itr in range(1, self.max_iter + 1):
            if self.optimizer_type == "bgd":
                X, y = X_train, y_train
                self._fit_handler(X, y)

            if self.optimizer_type == "sgd":
                indices_arr = np.arange(len(X_train))
                np.random.RandomState(self.random_state).shuffle(indices_arr)
                for idx in indices_arr:
                    X, y = X_train[idx:idx + 1], y_train[idx:idx + 1]
                    self._fit_handler(X, y)

            elif self.optimizer_type == "mbgd":
                indices_arr = np.arange(len(X_train))
                np.random.RandomState(self.random_state).shuffle(indices_arr)
                for idx in range(0, len(X_train), self.batch_size):
                    batch_indices = indices_arr[idx:idx + self.batch_size]
                    X, y = X_train[batch_indices], y_train[batch_indices]
                    self._fit_handler(X, y)

            if self.enable_logging:
                y_hat_train = self.predict(X_train)
                y_hat_validation = self.predict(X_validation)

                if self.objective == "classification":
                    wandb.log(
                        {'epoch': itr, 'train/loss': self._cost(X_train, y_train),
                         'train/accuracy': accuracy_score(y_train, y_hat_train),
                         'train/precision': precision_score(y_train, y_hat_train, average='micro'),
                         'train/recall': recall_score(y_train, y_hat_train, average='micro'),
                         'train/f1_score': f1_score(y_train, y_hat_train, average='micro'),
                         'validation/loss': self._cost(X_validation, y_validation),
                         'validation/accuracy': accuracy_score(y_validation, y_hat_validation),
                         'validation/precision': precision_score(y_validation, y_hat_validation, average='micro'),
                         'validation/recall': recall_score(y_validation, y_hat_validation, average='micro'),
                         'validation/f1_score': f1_score(y_validation, y_hat_validation, average='micro')})
                elif self.objective == "multi-label classification":
                    wandb.log(
                        {'epoch': itr, 'train/loss': self._cost(X_train, y_train),
                         'train/accuracy': accuracy_score(y_train, y_hat_train),
                         'train/precision': precision_score(y_train, y_hat_train, average='micro'),
                         'train/recall': recall_score(y_train, y_hat_train, average='micro'),
                         'train/f1_score': f1_score(y_train, y_hat_train, average='micro'),
                         'train/hamming_loss': hamming_loss(y_train, y_hat_train),
                         'validation/loss': self._cost(X_validation, y_validation),
                         'validation/accuracy': accuracy_score(y_validation, y_hat_validation),
                         'validation/precision': precision_score(y_validation, y_hat_validation, average='micro'),
                         'validation/recall': recall_score(y_validation, y_hat_validation, average='micro'),
                         'validation/f1_score': f1_score(y_validation, y_hat_validation, average='micro'),
                         'validation/hamming_loss': hamming_loss(y_validation, y_hat_validation)})
                elif self.objective == "regression":
                    wandb.log(
                        {'epoch': itr, 'train/loss': self._cost(X_train, y_train),
                         'train/MSE': mean_squared_error(y_train, y_hat_train),
                         'train/RMSE': root_mean_squared_error(y_train, y_hat_train),
                         'train/R2': r2_score(y_train, y_hat_train),
                         'validation/loss': self._cost(X_validation, y_validation),
                         'validation/MSE': mean_squared_error(y_validation, y_hat_validation),
                         'validation/RMSE': root_mean_squared_error(y_validation, y_hat_validation),
                         'validation/R2': r2_score(y_validation, y_hat_validation)})

            costs.append(self._cost(X_train, y_train))

            if self.tol and itr > 1:
                if abs(costs[-2] - costs[-1]) < self.tol:
                    if current_patience == self.patience:  # early stopping condition
                        break
                    else:
                        current_patience += 1
                else:
                    current_patience = 1

        return costs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        To predict class labels for the given input data, for either regression or classification.
        :param X: Input data
        :return: Predicted class labels
        """
        if self.objective == "classification" and self.loss_type is None:
            y_hat = self.forward_propagation(X)[0]
            predictions = np.zeros_like(y_hat)
            predictions[np.arange(len(y_hat)), y_hat.argmax(axis=1)] = 1
            return predictions
        elif self.objective == "multi-label classification" or self.loss_type is not None:
            y_hat = self.forward_propagation(X)[0]
            return (y_hat >= 0.5).astype(int)  # 0.5 probability threshold for label to be chosen
        elif self.objective == "regression":
            return self.forward_propagation(X)[0]

    def print_metrics(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> None:
        print("Metrics:")
        if self.objective == "classification":
            print(
                f"Loss {self._cost(X, y)}, Accuracy: {accuracy_score(y, y_hat)}, Precision (Micro): {precision_score(y, y_hat, average='micro')}, Precision (Macro): {precision_score(y, y_hat, average='macro')}, Recall (Micro): {recall_score(y, y_hat, average='micro')}, Recall (Macro): {recall_score(y, y_hat, average='macro')}, F1 Score (Micro): {f1_score(y, y_hat, average='micro')}, F1 Score (Macro): {f1_score(y, y_hat, average='macro')}")
        elif self.objective == "multi-label classification":
            print(
                f"Loss {self._cost(X, y)}, Accuracy: {accuracy_score(y, y_hat)}, Precision (Micro): {precision_score(y, y_hat, average='micro')}, Precision (Macro): {precision_score(y, y_hat, average='macro')}, Recall (Micro): {recall_score(y, y_hat, average='micro')}, Recall (Macro): {recall_score(y, y_hat, average='macro')}, F1 Score (Micro): {f1_score(y, y_hat, average='micro')}, F1 Score (Macro): {f1_score(y, y_hat, average='macro')}, Hamming Loss: {hamming_loss(y, y_hat)}")
        elif self.objective == "regression":
            print(
                f"Loss: {self._cost(X, y)}, MSE: {mean_squared_error(y, y_hat)}, MAE: {mean_absolute_error(y, y_hat)}, RMSE: {root_mean_squared_error(y, y_hat)}, R2: {r2_score(y, y_hat)}")

    def _compute_numerical_gradients(self, X: np.ndarray, y: np.ndarray, epsilon: float):
        num_grad_weights = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
        num_grad_biases = [np.zeros_like(self.biases[i]) for i in range(len(self.biases))]

        for i in range(len(self.weights)):
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    self.weights[i][j, k] += epsilon
                    cost_plus = self._cost(X, y)
                    self.weights[i][j, k] -= 2 * epsilon
                    cost_minus = self._cost(X, y)
                    num_grad_weights[i][j, k] = (cost_plus - cost_minus) / (2 * epsilon)
                    self.weights[i][j, k] += epsilon

            for j in range(self.biases[i].shape[1]):
                self.biases[i][0, j] += epsilon
                cost_plus = self._cost(X, y)
                self.biases[i][0, j] -= 2 * epsilon
                cost_minus = self._cost(X, y)
                num_grad_biases[i][0, j] = (cost_plus - cost_minus) / (2 * epsilon)
                self.biases[i][0, j] += epsilon

        return num_grad_weights, num_grad_biases

    def test_gradient_descent(self, X, y, epsilon: float = 1e-8, error_tolerance: float = 1e-6):
        num_grad_weights, num_grad_biases = self._compute_numerical_gradients(X, y, epsilon)

        y_hat, z_vals, activation_vals = self.forward_propagation(X)
        mlp_grad_weights, mlp_grad_biases = self.backward_propagation(X, y, y_hat, z_vals, activation_vals)

        for i in range(len(self.weights)):
            num_grad_flat = np.abs(num_grad_weights[i].flatten())
            backprop_grad_flat = np.abs(mlp_grad_weights[i].flatten())
            error = np.linalg.norm(num_grad_flat - backprop_grad_flat) / np.linalg.norm(
                num_grad_flat + backprop_grad_flat + 1e-15)

            if error > error_tolerance:
                print(f"Warning: Weight gradients do not match closely enough for layer {i + 1}! Error: {error}")
            else:
                print(f"Layer {i + 1} weights: Gradients match within the acceptable range.")

            num_grad_flat = np.abs(num_grad_biases[i].flatten())
            backprop_grad_flat = np.abs(mlp_grad_biases[i].flatten())
            error = np.linalg.norm(num_grad_flat - backprop_grad_flat) / np.linalg.norm(
                num_grad_flat + backprop_grad_flat + 1e-15)

            if error > error_tolerance:
                print(f"Warning: Bias gradients do not match closely enough for layer {i + 1}! Error: {error}")
            else:
                print(f"Layer {i + 1} biases: Gradients match within the acceptable range.")

    def save_model(self, file_path: str) -> None:
        with open(file_path, 'wb') as file:
            pickle.dump({'objective': self.objective, 'n_neurons_input_layer': self.n_neurons_input_layer,
                         'n_neurons_output_layer': self.n_neurons_output_layer, 'n_hidden_layers': self.n_hidden_layers,
                         'n_neurons_per_layer': self.n_neurons_per_layer, 'learning_rate': self.learning_rate,
                         'activation_function_chosen': self.activation_function_chosen,
                         'optimizer_type': self.optimizer_type, 'batch_size': self.batch_size,
                         'loss_type': self.loss_type, 'max_iter': self.max_iter, 'tol': self.tol,
                         'weights': self.weights, 'biases': self.biases, 'patience': self.patience}, file)

    def load_model(self, file_path: str) -> Self:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.objective = data['objective']
            self.n_hidden_layers = data['n_hidden_layers']
            self.n_neurons_per_layer = data['n_neurons_per_layer']
            self.learning_rate = data['learning_rate']
            self.activation_function_chosen = data['activation_function_chosen']
            self.activation = ActivationFunction(activation_function=self.activation_function_chosen)
            self.optimizer_type = data['optimizer_type']
            self.batch_size = data['batch_size']
            self.loss_type = data['loss_type']
            self.max_iter = data['max_iter']
            self.tol = data['tol']
            self.patience = data['patience']

            self.n_neurons_input_layer = data['n_neurons_input_layer']
            self.n_neurons_output_layer = data['n_neurons_output_layer']
            self.weights = data['weights']
            self.biases = data['biases']

        return self
