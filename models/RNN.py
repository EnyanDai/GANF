#%%
import torch
import torch.nn as nn
from functools import partial

class RecurrentEncoder(nn.Module):
    """Recurrent encoder"""

    def __init__(self, n_features, latent_dim, rnn):
        super().__init__()

        self.rec_enc1 = rnn(n_features, latent_dim, batch_first=True)

    def forward(self, x):
        _, h_n = self.rec_enc1(x)

        return h_n

class RecurrentDecoder(nn.Module):
    """Recurrent decoder for RNN and GRU"""

    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()

        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)

        # Squeezing
        h_i = h_0.squeeze()

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i)

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i)
            x = torch.cat([x, x_i], axis=1)

        return x.view(-1, seq_len, self.n_features)


class RecurrentDecoderLSTM(nn.Module):
    """Recurrent decoder LSTM"""

    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()

        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)

        # Squeezing
        h_i = [h.squeeze() for h in h_0]

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i[0])

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i[0])
            x = torch.cat([x, x_i], axis = 1)

        return x.view(-1, seq_len, self.n_features)


class RecurrentAE(nn.Module):
    """Recurrent autoencoder"""

    def __init__(self, n_features, latent_dim, device):
        super().__init__()

        # Encoder and decoder argsuration
        self.rnn, self.rnn_cell = nn.LSTM, nn.LSTMCell
        self.decoder = RecurrentDecoderLSTM
        self.latent_dim = latent_dim
        self.n_features = n_features
        self.device = device

        # Encoder and decoder
        self.encoder = RecurrentEncoder(self.n_features, self.latent_dim, self.rnn)
        self.decoder = self.decoder(self.latent_dim, self.n_features, self.rnn_cell, self.device)

    def forward(self, x):
        # x: N X K X L X D 
        seq_len = x.shape[1]
        h_n = self.encoder(x)
        out = self.decoder(h_n, seq_len)

        return torch.flip(out, [1])

# %%
