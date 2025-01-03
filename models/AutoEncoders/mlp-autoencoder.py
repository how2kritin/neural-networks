"""
Reference: https://scikit-neuralnetwork.readthedocs.io/en/latest/module_ae.html
"""
from models.MLP.MLP import MLP
import numpy as np


class AutoEncoders:
    def __init__(self, n_features: int, n_reduced_features: int, n_hidden_layers: int = 1,
                 n_neurons_per_layer: list[int] = [20], learning_rate: float = 0.001, activation: str = "sigmoid",
                 optimizer_type: str = 'bgd', batch_size: int = 32, max_iter: int = 200, random_state: int = None,
                 tol: float = None, patience: int = 1):
        """
        Initialize an AutoEncoder object. Uses my MLP for regression.
        :param n_features: Number of features in the input.
        :param n_reduced_features: Number of reduced features in the encoded layer, prior to decoding.
        :param n_hidden_layers: 1 by default. Number of hidden layers to use, UNTIL BEFORE THE REDUCTION LAYER.
        :param n_neurons_per_layer: [20] by default. List of integers specifying the number of neurons in each hidden layer, UNTIL BEFORE THE REDUCTION LAYER.
        :param learning_rate: 0.001 by default.
        :param activation: Sigmoid by default. Choose from {"sigmoid", "tanh", "relu", "linear"}.
        :param optimizer_type: 'bgd' by default. Choose from {"bgd", "mbgd", "sgd"}.
        :param batch_size: 32 by default. Number of batches, only applicable to optimizer_type == 'mbgd'.
        :param max_iter: 200 by default. Maximum number of iterations to train the neural network on.
        :param random_state: None by default. Used to deterministically initialize the MLP model's weights and biases.
        :param tol: None by default. Used for early stopping, if a value is set.
        :param patience: 1 by default. Number of consecutive iterations that the difference between successive loss is less than threshold, before early stopping is triggered.
        """
        n_neurons_per_layer = n_neurons_per_layer[:n_hidden_layers]  # remove extra vals if any
        n_neurons_per_layer_reversed = reversed(n_neurons_per_layer)
        n_neurons_per_layer.append(n_reduced_features)
        n_neurons_per_layer.extend(n_neurons_per_layer_reversed)

        self.n_features = n_features
        self.n_reduced_features = n_reduced_features
        self.n_hidden_layers = len(n_neurons_per_layer)
        self.n_neurons_per_layer = n_neurons_per_layer
        self.mlp = MLP(objective="regression", n_neurons_input_layer=self.n_features, n_neurons_output_layer=self.n_features,
                       n_hidden_layers=self.n_hidden_layers, n_neurons_per_layer=self.n_neurons_per_layer,
                       learning_rate_init=learning_rate, activation=activation, optimizer_type=optimizer_type,
                       batch_size=batch_size, random_state=random_state, max_iter=max_iter, tol=tol, patience=patience)

    def fit(self, X_train: np.ndarray) -> None:
        """
        Train the network. Essentially, you are training the MLP model. You are not fitting anything, so we don't need labels.
        :param X_train:
        :return:
        """
        self.mlp.fit(X_train, X_train)

    def get_latent(self, X: np.ndarray) -> np.ndarray:
        """
        Get the reduced dataset.
        :param X:
        :return:
        """
        activation_vals = self.mlp.forward_propagation(X)[2]
        return activation_vals[len(activation_vals) // 2]  # get the middle layer output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get the final reconstructed dataset. Essentially, this is the output of the MLP.
        :param X:
        :return:
        """
        return self.mlp.predict(X)

    def print_reconstruction_loss(self, X: np.ndarray) -> None:
        reconstructed_data = self.predict(X)
        reconstruction_error = np.mean(np.square(reconstructed_data - X))  # Mean Squared Error
        print(f"Reconstruction loss is {reconstruction_error}")