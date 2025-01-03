from typing import Callable, Optional, Literal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class RNN(nn.Module):
    """
    Ref: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    """

    def __init__(self, learning_rate: float, optimizer: Literal['sgd', 'adagrad', 'adamw'], n_features: int,
                 hidden_size: int, n_layers: int, dropout_rate: float = 0.0, n_epochs: int = 10,
                 enable_norm: bool = False, device: str = "cpu"):
        """
        RNN model implemented using LSTM layers.
        Includes a train_model() loop as well as a predict_individual() and predict_batch() methods for ease of use.

        :param optimizer: Choose an optimizer from {'sgd', 'adagrad', 'adamw'}.
        :param n_features: Number of expected features in the input.
        :param hidden_size: Number of features in the hidden state.
        :param n_layers: Number of RNN layers.
        :param dropout_rate: 0 by default. Set a value between 0 and 1 to enable dropout.
        :param n_epochs: Number of epochs this algorithm is supposed to run for in the training loop.
        :param enable_norm: False by default. Set to True to enable LayerNormalization.
        """
        super(RNN, self).__init__()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.device = device
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                           dropout=dropout_rate).to(self.device)
        self.enable_norm = enable_norm
        if self.enable_norm:
            self.norm = nn.LayerNorm(hidden_size).to(self.device)
        self.fc = nn.Linear(hidden_size, 1).to(self.device)
        self.optimizer_name = optimizer
        self._init_optimizer()
        self.loss_fn = nn.MSELoss().to(self.device)
        self.mae_fn = nn.L1Loss().to(self.device)
        self.to(self.device)

    def _init_optimizer(self) -> None:
        self.optimizer: Callable
        match self.optimizer_name:
            case 'sgd':
                self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
            case 'adagrad':
                self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
            case 'adamw':
                self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1, self.n_features)
        h_initial = torch.zeros(self.n_layers, x.size(0), self.hidden_size, device=self.device)
        c_initial = torch.zeros(self.n_layers, x.size(0), self.hidden_size, device=self.device)
        _, (hidden, _) = self.rnn(x, (h_initial, c_initial))
        output = hidden[-1]
        if self.enable_norm:
            output = self.norm(output)
        output = self.fc(output)
        return output.squeeze(-1)

    def train_model(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        for epoch in range(self.n_epochs):
            self.train()
            train_mse = 0.0
            train_mae = 0.0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.n_epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_mse += loss.item()
                train_mae += self.mae_fn(outputs, labels).item()

            train_mse /= len(train_loader)
            train_mae /= len(train_loader)

            if val_loader is not None:
                self.eval()
                val_mse = 0.0
                val_mae = 0.0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self(inputs)

                        val_mse += self.loss_fn(outputs, labels).item()
                        val_mae += self.mae_fn(outputs, labels).item()

                val_mse /= len(val_loader)
                val_mae /= len(val_loader)

                print(f'Epoch {epoch + 1}, Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}, '
                      f'Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}')
            else:
                print(f'Epoch {epoch + 1}, Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}')

    def predict_individual(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output = self(x)
        return output

    def predict_batch(self, dataloader: DataLoader):
        self.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in tqdm(dataloader, desc="Predicting on batch"):
                inputs = inputs.to(self.device)
                output = self(inputs)
                predictions.append(output.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        return predictions