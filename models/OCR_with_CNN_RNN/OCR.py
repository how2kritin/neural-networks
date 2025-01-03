import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class CNNRNNOCRModel(nn.Module):
    def __init__(self, n_epochs: int, vocab_size: int, max_sequence_length: int = 32,
                 learning_rate: float = 0.001, n_cnn_filters: list[int] = [32, 64, 128],
                 rnn_hidden_size: int = 256, rnn_n_layers: int = 2, device: str = "cpu"):
        super().__init__()
        self.n_epochs = n_epochs
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.n_cnn_filters = n_cnn_filters
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_n_layers = rnn_n_layers
        self.device = device
        self.max_sequence_length = max_sequence_length

        cnn_layers = []
        in_channels = 1
        for out_channels in n_cnn_filters:
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers).to(self.device)
        self.cnn_output_size = self._get_cnn_output_size()
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_n_layers,
            batch_first=True,
            bidirectional=True
        ).to(self.device)

        self.fc = nn.Linear(rnn_hidden_size * 2, vocab_size).to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        self.to(self.device)

    def _get_cnn_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 256)
            dummy_output = self.cnn(dummy_input)
            return dummy_output.shape[1] * dummy_output.shape[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        conv_out = self.cnn(x)
        conv_out = conv_out.permute(0, 2, 1, 3)
        conv_out = conv_out.reshape(batch_size, -1, conv_out.size(3))
        conv_out = conv_out.permute(0, 2, 1)

        rnn_out, _ = self.rnn(conv_out)

        output = self.fc(rnn_out)
        return output

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, idx_to_char):
        for epoch in range(self.n_epochs):
            self.train()
            total_loss = 0

            for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False, dynamic_ncols=True):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(batch_x)

                B, T, C = outputs.shape
                loss = self.criterion(
                    outputs.view(-1, C),
                    batch_y.view(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                self.optimizer.step()

                total_loss += loss.item()

            val_loss = self.evaluate(val_loader, idx_to_char)
            avg_train_loss = total_loss / len(train_loader)
            torch.save(model.state_dict(), 'cnn_rnn_bestmodel.pth')
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")

    def evaluate(self, loader: DataLoader, idx_to_char):
        self.eval()
        total_loss = 0
        total_correct_chars = 0
        total_chars = 0

        with torch.no_grad():
            for batch_x, batch_y in tqdm(loader, desc='Validating', leave=False):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self(batch_x)
                B, T, C = outputs.shape
                loss = self.criterion(
                    outputs.view(-1, C),
                    batch_y.view(-1)
                )
                total_loss += loss.item()

                predictions = outputs.argmax(dim=-1)
                for pred_indices, target_indices in zip(predictions, batch_y):
                    target = ''.join([idx_to_char[idx.item()] for idx in target_indices if idx.item() != 0])
                    target = target.split("<EOS>", 1)[0]

                    pred = ''.join([idx_to_char[idx.item()] for idx in pred_indices if idx.item() != 0])
                    pred = pred.split("<EOS>", 1)[0]

                    pred = pred.ljust(len(target))

                    correct_chars = sum(1 for p, t in zip(pred, target) if p == t and p != '<PAD>')
                    total_correct_chars += correct_chars
                    total_chars += len(target)

        val_loss = total_loss / len(loader)
        avg_correct_chars = total_correct_chars / total_chars if total_chars > 0 else 0

        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Average Correct Characters: {avg_correct_chars:.2%}")

        return val_loss, avg_correct_chars