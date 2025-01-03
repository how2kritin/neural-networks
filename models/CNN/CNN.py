from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb


class CNN(nn.Module):
    def __init__(self, task: str, input_dim: int, n_inp_channels: int, n_outputs: int, learning_rate: float,
                 dropout_rate: float, n_conv_layers: int, optimizer: str = 'adamw', activation: str = 'sigmoid',
                 n_epochs: int = 10, tol: float = 1e-4, patience: int = 3, round_outputs: bool = False,
                 enable_logging: bool = False,
                 get_regression_accuracy_for_evaluate: bool = False, random_state: int = 44):
        """
        :param task: Must be one of {'classification', 'regression', 'multi-class classification'}.
        :param optimizer: adamw by default. Must be one of {'sgd', 'adagrad', 'adamw'}.
        :param activation: sigmoid by default. Must be one of {'relu', 'sigmoid', 'tanh'}.
        :param n_inp_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        :param round_outputs: False by default. Determines whether the outputs must be rounded and reported for regression tasks.
        :param enable_logging: False by default. Used to enable logging in WandB.
        :param get_regression_accuracy_for_evaluate: False by default. Used to get accuracy instead of MSE when using evaluate() method for regression task.
        """
        super(CNN, self).__init__()
        self.task: str = task
        self.input_dim: int = input_dim
        self.n_inp_channels = n_inp_channels
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.n_conv_layers = n_conv_layers
        self.optimizer_name = optimizer
        self.activation_name = activation
        self.n_epochs = n_epochs
        self.tol = tol
        self.patience = patience
        self.round_outputs = round_outputs
        self.enable_logging = enable_logging
        self.get_regression_accuracy_for_evaluate = get_regression_accuracy_for_evaluate
        self.device: str = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
        self.training_history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [],
                                 'train_hamming_loss': [], 'val_hamming_loss': [], 'train_mse': [], 'val_mse': []}
        self.random_state = random_state

        self._init_activation()
        self._init_layers()
        self._init_optimizer_and_loss()
        self.feature_maps: list = []
        self.to(self.device)

    def _init_optimizer_and_loss(self) -> None:
        self.optimizer: Callable
        match self.optimizer_name:
            case 'sgd':
                self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
            case 'adagrad':
                self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
            case 'adamw':
                self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        self.loss_fn: Callable
        if self.task == 'classification':
            self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        elif self.task == 'regression':
            self.loss_fn = nn.MSELoss().to(self.device)
        elif self.task == 'multi-label classification':
            self.loss_fn = nn.BCELoss().to(self.device)

    def _init_activation(self) -> None:
        self.activation: Callable
        match self.activation_name:
            case 'relu':
                self.activation = nn.ReLU().to(self.device)
            case 'sigmoid':
                self.activation = nn.Sigmoid().to(self.device)
            case 'tanh':
                self.activation = nn.Tanh().to(self.device)

    def _init_layers(self) -> None:
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        current_channels = self.n_inp_channels
        for i in range(self.n_conv_layers):
            # double the number of output channels in each layer
            out_channels = 32 * (2 ** i)
            self.conv_layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
            self.batch_norm_layers.append(nn.BatchNorm2d(out_channels))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.activation_layers.append(self.activation)
            self.dropout_layers.append(nn.Dropout(self.dropout_rate))
            current_channels = out_channels

        # calc flattened features size
        final_spatial_dim = self.input_dim // (2 ** self.n_conv_layers)
        final_channels = 32 * (2 ** (self.n_conv_layers - 1))
        flattened_size = final_channels * (final_spatial_dim ** 2)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.BatchNorm1d(512),
            self.activation,
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            self.activation,
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, self.n_outputs),
        )

        # output activation layer
        if self.task == 'classification':
            self.activation_layers.append(nn.Softmax(dim=1))
        elif self.task == 'multi-label classification':
            self.activation_layers.append(nn.Sigmoid())
        elif self.task == 'regression':
            self.activation_layers.append(nn.Identity())

        self.fc_layers.to(self.device)
        self.conv_layers.to(self.device)
        self.batch_norm_layers.to(self.device)
        self.activation_layers.to(self.device)
        self.pool_layers.to(self.device)
        self.dropout_layers.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps_batch = []

        # pass input through convolution layers.
        feature_maps_batch.append(x)
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.batch_norm_layers[i](x)
            x = self.activation_layers[i](x)
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)
            feature_maps_batch.append(x.detach().cpu())  # store feature maps

        self.feature_maps.append(feature_maps_batch)

        # flatten the output from convolutional layers, and pass it through the fully connected layers.
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        if self.task == 'regression':
            x = self.activation_layers[-1](x)
            return x.squeeze()
        else:
            return self.activation_layers[-1](x)

    def train_model(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        best_val_loss = float('inf')
        curr_patience = 0

        for epoch in range(self.n_epochs):
            self.train()
            total_train_loss = torch.tensor(0.0, device=self.device)
            total_train_correct = 0
            total_train_samples = 0
            total_train_mse = 0.0
            total_hamming_loss = 0.0

            # training phase
            for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.n_epochs}'):
                batch_x = batch_x.to(self.device)
                if self.task == 'regression':
                    batch_y = batch_y.float().to(self.device)
                    if len(batch_y.shape) == 1:
                        batch_y = batch_y.unsqueeze(1)
                else:
                    batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(batch_x)

                if self.task == 'regression':
                    if len(outputs.shape) == 1:
                        outputs = outputs.unsqueeze(1)
                    loss = self.loss_fn(outputs, batch_y)
                else:
                    loss = self.loss_fn(outputs, batch_y)

                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

                if self.task == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    total_train_correct += (predicted == batch_y).sum().item()
                elif self.task == 'multi-label classification':
                    predicted = (outputs > 0.5).float()
                    total_train_correct += (predicted == batch_y).sum().item()
                    hamming_loss = ((predicted != batch_y).sum(dim=1).float() / batch_y.size(1)).sum().item()
                    total_hamming_loss += hamming_loss
                elif self.task == 'regression':
                    mse = ((outputs - batch_y) ** 2).mean().item()
                    total_train_mse += mse * batch_x.size(0)

                total_train_samples += batch_x.size(0)

            avg_train_loss = (total_train_loss / len(train_loader)).item()
            self.training_history['train_loss'].append(avg_train_loss)

            if self.task == 'classification':
                train_accuracy = total_train_correct / total_train_samples if total_train_samples > 0 else 0.0
                self.training_history['train_accuracy'].append(train_accuracy)
                #                 print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.3f}, Train Accuracy = {train_accuracy:.3f}')
                if self.enable_logging:
                    wandb.log({'epoch': epoch + 1, 'train/accuracy': train_accuracy, 'train/loss': avg_train_loss})
            elif self.task == 'multi-label classification':
                train_accuracy = total_train_correct / (total_train_samples * batch_y.size(1))
                train_hamming_loss = total_hamming_loss / total_train_samples
                self.training_history['train_accuracy'].append(train_accuracy)
                self.training_history['train_hamming_loss'].append(train_hamming_loss)
                #                 print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.3f}, Train Exact Match Accuracy = {train_accuracy:.3f}, Train Hamming Loss = {train_hamming_loss:.3f}')
                if self.enable_logging:
                    wandb.log({'epoch': epoch + 1, 'train/exact_match_accuracy': train_accuracy,
                               'train/hamming_loss': train_hamming_loss, 'train/loss': avg_train_loss})
            elif self.task == 'regression':
                train_mse = total_train_mse / total_train_samples if total_train_samples > 0 else 0.0
                self.training_history['train_mse'].append(train_mse)
                #                 print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.3f}, Train MSE = {train_mse:.3f}')
                if self.enable_logging:
                    wandb.log({'epoch': epoch + 1, 'train/MSE': train_mse, 'train/loss': avg_train_loss})

            # validation phase
            if val_loader is not None:
                val_loss, val_accuracy, val_hamming_loss = self.evaluate(val_loader)
                self.training_history['val_loss'].append(val_loss)

                if self.task == 'classification':
                    self.training_history['val_accuracy'].append(val_accuracy)
                    #                     print(f'Epoch {epoch + 1}: Validation Loss = {val_loss:.3f}, Validation Accuracy = {val_accuracy:.3f}')
                    if self.enable_logging:
                        wandb.log(
                            {'epoch': epoch + 1, 'validation/accuracy': val_accuracy, 'validation/loss': val_loss})
                elif self.task == 'multi-label classification':
                    self.training_history['val_accuracy'].append(val_accuracy)
                    self.training_history['val_hamming_loss'].append(val_hamming_loss)
                    #                     print(f'Epoch {epoch + 1}: Validation Loss = {val_loss:.3f}, Validation Exact Match Accuracy = {val_accuracy:.3f}, Validation Hamming Loss = {val_hamming_loss:.3f}')
                    if self.enable_logging:
                        wandb.log({'epoch': epoch + 1, 'validation/exact_match_accuracy': val_accuracy,
                                   'validation/hamming_loss': val_hamming_loss, 'validation/loss': val_loss})
                elif self.task == 'regression':
                    self.training_history['val_mse'].append(
                        val_accuracy)  # treating `val_accuracy` as MSE in regression (this is what self.evaluate() does)
                    #                     print(f'Epoch {epoch + 1}: Validation Loss = {val_loss:.3f}, Validation MSE = {val_accuracy:.3f}')
                    if self.enable_logging:
                        wandb.log({'epoch': epoch + 1, 'validation/MSE': val_accuracy, 'validation/loss': val_loss})

                # check for early stopping
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    curr_patience = 0
                else:
                    curr_patience += 1

                if curr_patience >= self.patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    break

    def evaluate(self, data_loader: DataLoader) -> tuple[float, float, float]:
        self.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = 0
        total_samples = 0
        total_exact_matches = 0
        total_label_matches = 0
        total_labels = 0
        total_mse = 0.0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                if self.task == 'regression':
                    batch_y = batch_y.float().to(self.device)
                    if len(batch_y.shape) == 1:
                        batch_y = batch_y.unsqueeze(1)
                else:
                    batch_y = batch_y.to(self.device)

                outputs = self(batch_x)
                if self.task == 'regression':
                    if len(outputs.shape) == 1:
                        outputs = outputs.unsqueeze(1)
                    if self.round_outputs:
                        outputs = torch.round(outputs.float())
                loss = self.loss_fn(outputs, batch_y)
                total_loss += loss.item()

                if self.task == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == batch_y).sum().item()
                elif self.task == 'multi-label classification':
                    predicted = (outputs > 0.5).float()
                    exact_matches = (predicted == batch_y).all(dim=1).sum().item()
                    total_exact_matches += exact_matches
                    label_matches = (predicted == batch_y).sum().item()
                    total_label_matches += label_matches
                    total_labels += batch_y.numel()
                elif self.task == 'regression' and not self.get_regression_accuracy_for_evaluate:
                    mse = ((outputs - batch_y) ** 2).mean().item()
                    total_mse += mse * batch_x.size(0)
                elif self.task == 'regression' and self.get_regression_accuracy_for_evaluate:
                    total_correct += (outputs == batch_y).sum().item()

                total_samples += batch_x.size(0)

        avg_loss = (total_loss / len(data_loader)).item()
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        if self.task == 'multi-label classification':
            exact_match_accuracy = total_exact_matches / total_samples if total_samples > 0 else 0.0
            label_wise_accuracy = total_label_matches / total_labels if total_labels > 0 else 0.0
            hamming_loss = 1.0 - label_wise_accuracy
            return avg_loss, exact_match_accuracy, hamming_loss
        elif self.task == 'classification' or (self.task == 'regression' and self.get_regression_accuracy_for_evaluate):
            return avg_loss, accuracy, 0.0
        elif self.task == 'regression' and not self.get_regression_accuracy_for_evaluate:
            mse = total_mse / total_samples if total_samples > 0 else 0.0
            return avg_loss, mse, 0.0

    def visualize_feature_maps(self, max_num_figures: int = 3) -> None:
        self.eval()
        n_figures_mapped = 0

        for batch in self.feature_maps:
            if n_figures_mapped >= max_num_figures:
                break

            batch_size = batch[0].shape[0]
            for figure_idx in range(batch_size):
                if n_figures_mapped >= max_num_figures:
                    break

                plt.figure(figsize=(10, 10))

                for layer_idx, feature_map in enumerate(batch):
                    plt.subplot(2, 2, layer_idx + 1)
                    if layer_idx == 0:
                        img = feature_map[figure_idx, 0].cpu()
                        plt.imshow(img, cmap='gray')
                        plt.title("Original Input")
                    else:
                        feat = feature_map[figure_idx, 0].cpu()
                        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
                        plt.imshow(feat, cmap='viridis')
                        plt.title(f"Layer {layer_idx}")
                    plt.axis('off')

                plt.tight_layout()
                plt.show()
                n_figures_mapped += 1

    def plot_training_and_validation_loss_history(self, learning_rate: float, dropout_rate: float, n_conv_layers: str,
                                                  optimizer: str, activation: str) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        ax1.plot(epochs, self.training_history['train_loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2.plot(epochs, self.training_history['val_loss'])
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        fig.suptitle(
            f"Learning Rate: {learning_rate}, Dropout: {dropout_rate}, Num Convolution Layers: {n_conv_layers}, Optimizer: {optimizer}, Activation: {activation}")
        plt.savefig(
            f"hyperparameter_tuning_loss_plots_lr={learning_rate}_dr={dropout_rate}_cl={n_conv_layers}_opt={optimizer}_act={activation}.png")
        plt.close()