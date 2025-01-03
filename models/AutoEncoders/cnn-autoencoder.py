from typing import Optional, Literal
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib
from typing import Callable
import numpy as np


class CnnAutoencoder(nn.Module):
    def __init__(self, learning_rate: float, conv_kernel_size: int, n_inp_channels: int,
                 num_filters_per_layer: list[int], n_epochs: int = 5, patience: int = 2,
                 optimizer: Literal['sgd', 'adagrad', 'adamw'] = 'adamw',
                 enable_logging: bool = False, random_state: int = 44):
        """
        Use the train_model() method to train the model (training loop has been provided for ease of use).
        Includes methods to plot the latent space, to compare original and reconstructed images, and also to plot the variation in training and validation loss.

        :param n_inp_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        :param num_filters_per_layer: List of integers that defines the number of filters in each layer of the encoder. This will be reversed for the decoder.
        :param optimizer: adamw by default. Must be one of {'sgd', 'adagrad', 'adamw'}.
        :param enable_logging: False by default. Used to enable logging in WandB.
        """
        super(CnnAutoencoder, self).__init__()
        self.learning_rate = learning_rate
        self.conv_kernel_size = conv_kernel_size
        self.n_inp_channels = n_inp_channels
        self.num_filters_per_layer = num_filters_per_layer
        self.n_epochs = n_epochs
        self.patience = patience
        self.enable_logging = enable_logging
        self.training_history = {'train_loss': [], 'val_loss': []}
        self.optimizer_name = optimizer
        self.device: str = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
        self._init_layers()
        self._init_optimizer()
        self.random_state = random_state
        self.loss_fn = nn.MSELoss().to(self.device)
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

    def _calculate_padding(self, input_dim, kernel_size, stride=1):
        return (stride * (input_dim - 1) - input_dim + kernel_size) // 2

    def _init_layers(self) -> None:
        # encoder layers
        self.encoder_conv_layers = nn.ModuleList()
        self.encoder_pool_layers = nn.ModuleList()
        self.encoder_activation_layers = nn.ModuleList()

        current_channels = self.n_inp_channels
        for channel_size in self.num_filters_per_layer:
            padding = self._calculate_padding(input_dim=current_channels, kernel_size=self.conv_kernel_size)
            self.encoder_conv_layers.append(
                nn.Conv2d(current_channels, channel_size, kernel_size=self.conv_kernel_size, stride=1, padding=padding))
            self.encoder_activation_layers.append(nn.ReLU())
            self.encoder_pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            current_channels = channel_size

        # decoder layers
        reversed_channel_sizes = list(reversed(self.num_filters_per_layer))

        self.decoder_unpool_layers = nn.ModuleList()
        self.decoder_conv_layers = nn.ModuleList()
        self.decoder_activation_layers = nn.ModuleList()

        current_channels = reversed_channel_sizes[0]
        for i in range(len(reversed_channel_sizes)):
            out_channels = reversed_channel_sizes[i + 1] if i < len(reversed_channel_sizes) - 1 else self.n_inp_channels
            padding = self._calculate_padding(input_dim=current_channels, kernel_size=self.conv_kernel_size)
            self.decoder_unpool_layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
            self.decoder_conv_layers.append(
                nn.Conv2d(current_channels, out_channels, kernel_size=self.conv_kernel_size, stride=1,
                          padding=padding))
            # sigmoid for the last layer, relu for others
            self.decoder_activation_layers.append(nn.Sigmoid() if i == len(reversed_channel_sizes) - 1 else nn.ReLU())
            current_channels = out_channels

        self.encoder_conv_layers.to(self.device)
        self.encoder_pool_layers.to(self.device)
        self.encoder_activation_layers.to(self.device)
        self.decoder_unpool_layers.to(self.device)
        self.decoder_conv_layers.to(self.device)
        self.decoder_activation_layers.to(self.device)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list, list]:
        indices_list = []
        shapes_list = []

        for i in range(len(self.encoder_conv_layers)):
            x = self.encoder_conv_layers[i](x)
            x = self.encoder_activation_layers[i](x)
            shapes_list.append(x.shape)
            x, indices = self.encoder_pool_layers[i](x)
            indices_list.append(indices)

        return x, indices_list, shapes_list

    def decode(self, x: torch.Tensor, indices_list: list, shapes_list: list) -> torch.Tensor:
        indices_list = list(reversed(indices_list))
        shapes_list = list(reversed(shapes_list))

        for i in range(len(self.decoder_conv_layers)):
            x = self.decoder_unpool_layers[i](x, indices_list[i], output_size=shapes_list[i])
            x = self.decoder_conv_layers[i](x)
            x = self.decoder_activation_layers[i](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(*self.encode(x))

    def train_model(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """
        A handy training loop so that you don't have to write your own. You can either use this, or use the encode() and decode() methods to train this model.
        """
        best_val_loss = float('inf')
        curr_patience = 0

        for epoch in range(self.n_epochs):
            self.train()
            total_train_loss = torch.tensor(0.0, device=self.device)

            # training phase
            for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.n_epochs}'):
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(batch_x)
                loss = self.loss_fn(outputs, batch_x)  # loss is checked with the original inputs itself.
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = (total_train_loss / len(train_loader)).item()
            self.training_history['train_loss'].append(avg_train_loss)

            if self.enable_logging:
                wandb.log({'epoch': epoch + 1, 'train/loss': avg_train_loss})
            else:
                print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}')

            # validation phase
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.training_history['val_loss'].append(val_loss)

                if self.enable_logging:
                    wandb.log({'epoch': epoch + 1, 'validation/loss': val_loss})
                else:
                    print(f'Epoch {epoch + 1}: Validation Loss = {val_loss:.6f}')

                # check for early stopping
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    curr_patience = 0
                else:
                    curr_patience += 1

                if curr_patience >= self.patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    break

    def evaluate(self, data_loader: DataLoader) -> float:
        self.eval()
        total_loss = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self(batch_x)
                loss = self.loss_fn(outputs, batch_x)
                total_loss += loss.item()

        avg_loss = (total_loss / len(data_loader)).item()
        return avg_loss

    def plot_to_compare_original_and_reconstructed_images(self, image_batch: list, num_images: int = 10) -> None:
        self.eval()
        fig, axes = plt.subplots(2, num_images, figsize=(15, 6))

        with torch.no_grad():
            image_batch = image_batch.to(self.device)
            reconstructed_images = self.forward(image_batch)

            test_images_np = image_batch.cpu().detach().numpy()
            reconstructed_images_np = reconstructed_images.cpu().detach().numpy()
            reconstruction_error = np.mean(np.square(reconstructed_images_np - test_images_np))
            print("Reconstruction Error (on the first batch of images):", reconstruction_error)

            for i in range(num_images):
                axes[0, i].imshow(test_images_np[i].squeeze(), cmap='gray')
                axes[0, i].set_title(f'Original')
                axes[0, i].axis('off')

                axes[1, i].imshow(reconstructed_images_np[i].squeeze(), cmap='gray')
                axes[1, i].set_title(f'Reconstructed')
                axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

    def get_latent_space(self, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        encodings = []
        labels = []

        self.eval()
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                encoding = self.encode(batch_x)[0]
                encoding = encoding.view(encoding.size(0), -1)

                encodings.append(encoding.cpu().numpy())
                labels.append(batch_y.numpy())

        encodings = np.concatenate(encodings, axis=0)
        labels = np.concatenate(labels, axis=0)

        return encodings, labels

    def plot_latent_space(self, data_loader: DataLoader, max_num_entries_per_label: int = 10000) -> None:
        encodings, labels = self.get_latent_space(data_loader)

        selected_encodings = []
        selected_labels = []

        for label in range(10):
            indices = np.where(labels == label)[0]
            selected_indices = np.random.choice(indices, size=min(max_num_entries_per_label, len(indices)),
                                                replace=False)
            selected_encodings.append(encodings[selected_indices])
            selected_labels.append(labels[selected_indices])

        encodings = np.concatenate(selected_encodings, axis=0)
        labels = np.concatenate(selected_labels, axis=0)

        encodings_standardized = MinMaxScaler().fit_transform(encodings)

        pca_2d = PCA(n_components=2)
        pca_3d = PCA(n_components=3)

        encodings_2d = pca_2d.fit_transform(encodings_standardized)
        encodings_3d = pca_3d.fit_transform(encodings_standardized)

        plt.figure(figsize=(12, 5))
        plt.subplot(121)

        discrete_colors = [
            '#1F77B4',  # blue
            '#FF7F0E',  # orange
            '#2CA02C',  # green
            '#D62728',  # red
            '#9467BD',  # purple
            '#8C564B',  # brown
            '#E377C2',  # pink
            '#7F7F7F',  # gray
            '#BCBD22',  # olive green
            '#17BECF'  # cyan
        ]

        # 2D plot
        scatter = plt.scatter(encodings_2d[:, 0], encodings_2d[:, 1],
                              c=labels, cmap=matplotlib.colors.ListedColormap(discrete_colors),
                              alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter)
        plt.title('2D PCA')
        plt.xlabel(f'PC1 (Var: {pca_2d.explained_variance_ratio_[0] * 100:.2f}%)')
        plt.ylabel(f'PC2 (Var: {pca_2d.explained_variance_ratio_[1] * 100:.2f}%)')

        # 3D Plot
        ax = plt.subplot(122, projection='3d')
        scatter = ax.scatter(encodings_3d[:, 0], encodings_3d[:, 1], encodings_3d[:, 2],
                             c=labels, cmap=matplotlib.colors.ListedColormap(discrete_colors),
                             alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter)
        ax.set_title('3D PCA')
        ax.set_xlabel(f'PC1 (Var: {pca_3d.explained_variance_ratio_[0] * 100:.2f}%)')
        ax.set_ylabel(f'PC2 (Var: {pca_3d.explained_variance_ratio_[1] * 100:.2f}%)')
        ax.set_zlabel(f'PC3 (Var: {pca_3d.explained_variance_ratio_[2] * 100:.2f}%)')

        plt.tight_layout()
        plt.show()

    def plot_training_and_validation_loss_history(self) -> None:
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
            f"Learning Rate: {self.learning_rate}, Size of Convolution Kernel: {self.conv_kernel_size}, Num Filters Per Conv Layer: {str(self.num_filters_per_layer)}, Optimizer: {self.optimizer_name}")
        plt.savefig(
            f"hyperparameter_tuning_loss_plots_lr={self.learning_rate}_cks={self.conv_kernel_size}_nfcl={str(self.num_filters_per_layer)}_opt={self.optimizer_name}.png")
        plt.close()
