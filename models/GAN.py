#%%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer
def ConvEncoder(activation = nn.LeakyReLU, in_channels:int = 3, n_c:int = 64,
                    k_size:int = 5):
    
    enc = nn.Sequential(*(nn.Conv1d(in_channels, n_c, k_size, stride=2, padding=2),
                                            nn.BatchNorm1d(n_c),
                                            activation(),
                                            nn.Conv1d(n_c, n_c*2, k_size, stride=2, padding=2),
                                            nn.BatchNorm1d(n_c*2),
                                            activation(),
                                            nn.Conv1d(n_c*2, n_c*4, k_size, stride=2, padding=2),
                                            nn.BatchNorm1d(n_c*4),
                                            activation()))
    return enc

def ConvDecoder(activation = nn.LeakyReLU, in_channels:int = 3, n_c:int = 64,
                    k_size:int = 5):

    decoder = nn.Sequential(*(nn.ConvTranspose1d(n_c*4, n_c*2, k_size, stride=2, padding=2, output_padding=0),
                                            torch.nn.BatchNorm1d(n_c*2),
                                            activation(),
                                            torch.nn.ConvTranspose1d(n_c*2, n_c, k_size,stride=2, padding=2, output_padding=1),
                                            torch.nn.BatchNorm1d(n_c),
                                            activation(),
                                            torch.nn.ConvTranspose1d(n_c, in_channels, k_size,stride=2, padding=2, output_padding=1)))
    return decoder

class CNNAE(torch.nn.Module):
    """Recurrent autoencoder"""

    def __init__(self,in_channels:int = 3, n_channels:int = 16,
                    kernel_size:int = 5):
        super(CNNAE, self).__init__()

        # Encoder and decoder argsuration
        activation = torch.nn.LeakyReLU
        self.in_channels = in_channels
        self.n_c = n_channels
        self.k_size = kernel_size

        self.encoder = ConvEncoder(activation, in_channels, n_channels, kernel_size)

        self.decoder = ConvDecoder(activation, in_channels, n_channels, kernel_size)


    def forward(self, x:torch.Tensor):

        z = self.encoder.forward(x)

        x_out = self.decoder.forward(z)

        return x_out

class R_Net(torch.nn.Module):

    def __init__(self, activation = torch.nn.LeakyReLU, in_channels:int = 3, n_channels:int = 16,
                    kernel_size:int = 5, std:float = 0.2):

        super(R_Net, self).__init__()

        self.activation = activation
        self.in_channels = in_channels
        self.n_c = n_channels
        self.k_size = kernel_size
        self.std = std

        self.Encoder = ConvEncoder(activation, in_channels, n_channels, kernel_size)

        self.Decoder = ConvDecoder(activation, in_channels, n_channels, kernel_size)

    def forward(self, x:torch.Tensor, noise:bool = True):

        x_hat = self.add_noise(x) if noise else x
        z = self.Encoder.forward(x_hat)

        x_out = self.Decoder.forward(z)

        return x_out

    def add_noise(self, x):

        noise = torch.randn_like(x) * self.std
        x_hat = x + noise

        return x_hat

class D_Net(torch.nn.Module):

    def __init__(self, in_resolution:int, activation = torch.nn.LeakyReLU, in_channels:int = 3, n_channels:int = 16, kernel_size:int = 5):

        super(D_Net, self).__init__()

        self.activation = activation
        self.in_resolution = in_resolution
        self.in_channels = in_channels
        self.n_c = n_channels
        self.k_size = kernel_size

        self.cnn = ConvEncoder(activation, in_channels, n_channels, kernel_size)

        # Compute output dimension after conv part of D network

        self.out_dim = self._compute_out_dim()

        self.fc = torch.nn.Linear(self.out_dim, 1)

    def _compute_out_dim(self):
        
        test_x = torch.Tensor(1, self.in_channels, self.in_resolution)
        for p in self.cnn.parameters():
            p.requires_grad = False
        test_x = self.cnn(test_x)
        out_dim = torch.prod(torch.tensor(test_x.shape[1:])).item()
        for p in self.cnn.parameters():
            p.requires_grad = True

        return out_dim

    def forward(self, x:torch.Tensor):

        x = self.cnn(x)

        x = torch.flatten(x, start_dim = 1)

        out = self.fc(x)

        return out

def R_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float) -> dict:

    pred = d_net(x_fake)
    y = torch.ones_like(pred)

    rec_loss = F.mse_loss(x_fake, x_real)
    gen_loss = F.binary_cross_entropy_with_logits(pred, y) # generator loss

    L_r = gen_loss + lambd * rec_loss

    return {'rec_loss' : rec_loss, 'gen_loss' : gen_loss, 'L_r' : L_r}

def D_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:

    pred_real = d_net(x_real)
    pred_fake = d_net(x_fake.detach())
    
    y_real = torch.ones_like(pred_real)
    y_fake = torch.zeros_like(pred_fake)

    real_loss = F.binary_cross_entropy_with_logits(pred_real, y_real)
    fake_loss = F.binary_cross_entropy_with_logits(pred_fake, y_fake)

    return real_loss + fake_loss

# Wasserstein GAN loss (https://arxiv.org/abs/1701.07875)

def R_WLoss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float) -> dict:

    pred = torch.sigmoid(d_net(x_fake))

    rec_loss = F.mse_loss(x_fake, x_real)
    gen_loss = -torch.mean(pred) # Wasserstein G loss: - E[ D(G(x)) ]

    L_r = gen_loss + lambd * rec_loss

    return {'rec_loss' : rec_loss, 'gen_loss' : gen_loss, 'L_r' : L_r}

def D_WLoss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:

    pred_real = torch.sigmoid(d_net(x_real))
    pred_fake = torch.sigmoid(d_net(x_fake.detach()))
    
    dis_loss = -torch.mean(pred_real) + torch.mean(pred_fake) # Wasserstein D loss: -E[D(x_real)] + E[D(x_fake)]

    return dis_loss

# %%
def train_model(r_net: torch.nn.Module,
                d_net: torch.nn.Module,
                train_dataset: torch.utils.data.Dataset,
                valid_dataset: torch.utils.data.Dataset,
                r_loss = R_Loss,
                d_loss = D_Loss,
                lr_scheduler = None,
                optimizer_class = torch.optim.Adam,
                optim_r_params: dict = {},
                optim_d_params: dict = {},
                learning_rate: float = 0.001,
                scheduler_r_params: dict = {},
                scheduler_d_params: dict = {},
                batch_size: int = 1024,
                max_epochs: int = 40,
                epoch_step: int = 1,
                save_step: int = 5,
                lambd: float = 0.2,
                device: torch.device = torch.device('cuda'),
                save_path: str = ".") -> tuple:

    optim_r = optimizer_class(r_net.parameters(), lr = learning_rate, **optim_r_params)
    optim_d = optimizer_class(d_net.parameters(), lr = learning_rate, **optim_d_params)

    if lr_scheduler:
        scheduler_r = lr_scheduler(optim_r, **scheduler_r_params)
        scheduler_d = lr_scheduler(optim_d, **scheduler_d_params)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    for epoch in range(max_epochs):

        start = timer()
        train_metrics = train_single_epoch(r_net, d_net, optim_r, optim_d, r_loss, d_loss, train_loader, lambd, device)
        valid_metrics = validate_single_epoch(r_net, d_net, r_loss, d_loss, valid_loader, device)
        time = timer() - start


        if epoch % epoch_step == 0:
            print(f'Epoch {epoch}:')
            print('Train Metrics:', train_metrics)
            print('Val Metrics:', valid_metrics)
            print(f'TIME: {time:.2f} s')

        if lr_scheduler:
            scheduler_r.step()
            scheduler_d.step()

        if epoch % save_step == 0:
            torch.save(r_net.state_dict(), os.path.join(save_path, "r_net_{}.pt".format(epoch)))
            torch.save(d_net.state_dict(), os.path.join(save_path, "d_net_{}.pt".format(epoch)))
            print(f'Saving model on epoch {epoch}')

    return (r_net, d_net)

def train_single_epoch(r_net, d_net, optim_r, optim_d, r_loss, d_loss, train_loader, lambd, device) -> dict:

    r_net.train()
    d_net.train()

    train_metrics = {'rec_loss' : 0, 'gen_loss' : 0, 'dis_loss' : 0}

    for data in train_loader:

        x = data.to(device)
        x = torch.transpose(x, dim0=2, dim1=3)
        x_real = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        
        x_fake = r_net(x_real)
        d_net.zero_grad()

        dis_loss = d_loss(d_net, x_real, x_fake)

        dis_loss.backward()
        optim_d.step()

        r_net.zero_grad()

        r_metrics = r_loss(d_net, x_real, x_fake, lambd) # L_r = gen_loss + lambda * rec_loss

        r_metrics['L_r'].backward()
        optim_r.step()

        train_metrics['rec_loss'] += r_metrics['rec_loss']
        train_metrics['gen_loss'] += r_metrics['gen_loss']
        train_metrics['dis_loss'] += dis_loss

    train_metrics['rec_loss'] = train_metrics['rec_loss'].item() / (len(train_loader.dataset) / train_loader.batch_size)
    train_metrics['gen_loss'] = train_metrics['gen_loss'].item() / (len(train_loader.dataset) / train_loader.batch_size)
    train_metrics['dis_loss'] = train_metrics['dis_loss'].item() / (len(train_loader.dataset) / train_loader.batch_size)

    return train_metrics

def validate_single_epoch(r_net, d_net, r_loss, d_loss, valid_loader, device) -> dict:

    r_net.eval()
    d_net.eval()

    valid_metrics = {'rec_loss' : 0, 'gen_loss' : 0, 'dis_loss' : 0}

    with torch.no_grad():
        for data in valid_loader:

            x = data.to(device)
            x = torch.transpose(x, dim0=2, dim1=3)
            x_real = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])

            x_fake = r_net(x_real)

            dis_loss = d_loss(d_net, x_real, x_fake)

            r_metrics = r_loss(d_net, x_real, x_fake, 0)
                
            valid_metrics['rec_loss'] += r_metrics['rec_loss']
            valid_metrics['gen_loss'] += r_metrics['gen_loss']
            valid_metrics['dis_loss'] += dis_loss

        valid_metrics['rec_loss'] = valid_metrics['rec_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
        valid_metrics['gen_loss'] = valid_metrics['gen_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
        valid_metrics['dis_loss'] = valid_metrics['dis_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)

        return valid_metrics



# %%
