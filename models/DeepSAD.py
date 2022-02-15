
#%%
import json
import torch
import logging
import time
import torch

import torch.optim as optim

class AETrainer:

    def __init__(self, device: str = 'cuda'):

        self.device = device

    def train(self, train_loader, ae_net, args):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Training
        print('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(10):


            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:

                if isinstance(data, list):
                    data = data[0]
                
                x = data.to(self.device)
                x = torch.transpose(x, dim0=2, dim1=3)
                inputs = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, 10, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        print('Pretraining time: %.3f' % pretrain_time)
        print('Finished pretraining.')

        return ae_net

class DeepSVDDTrainer:

    def __init__(self, device: str = 'cuda'):

        self.device = device
        # Deep SVDD parameters
       
        self.c = None


    def train(self, train_loader, net, args):
        self.args = args

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            print(self.c.shape)
            print('Center c initialized.')

        # Training
        print('Starting training...')
        start_time = time.time()
        net.train()

        save_path = os.path.join(args.output_dir,args.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for epoch in range(args.n_epochs):

            scheduler.step()

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for data in train_loader:
                x = data.to(self.device)
                x = torch.transpose(x, dim0=2, dim1=3)
                inputs = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs).squeeze().mean(dim=-1)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, args.n_epochs, epoch_train_time, loss_epoch / n_batches))
            torch.save({'c': self.c, 'net_dict': net.state_dict()}, os.path.join(save_path, "{}_{}.pt".format(args.name, epoch)))
        self.train_time = time.time() - start_time
        print('Training time: %.3f' % self.train_time)

        print('Finished training.')

        return net

    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = 0.0

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                x = data.to(self.device)
                x = torch.transpose(x, dim0=2, dim1=3)
                inputs = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
                outputs = net(inputs).squeeze() 
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c.mean(dim=-1)


from models.RNN import RecurrentAE
from models.GAN import CNNAE


class DeepSVDD(object):

    def __init__(self, n_features, hidden_size, device):

        self.c = None  # hypersphere center c


        self.trainer = None

        # if encoder=='RNN':
        # self.ae_net = RecurrentAE(n_features, hidden_size, device)
        self.ae_net = CNNAE(n_features, hidden_size).to(device)
        self.net = self.ae_net.encoder

        self.ae_trainer = None
        self.results = {
            'test_auc': None
        }

    def train(self, dataset, args, device: str = 'cuda'):
        """Trains the Deep SVDD model on the training data."""

        self.trainer = DeepSVDDTrainer(device=device)
        # Get the model
        self.trainer.train(dataset, self.net, args)
        self.c = self.trainer.c

    def test(self, test_loader, delta_t, sigma, device):
        from utils import roc_auc_all
        import numpy as np
        self.net.eval()
        self.net.to(device)
        loss = []

        with torch.no_grad():
            for data in test_loader:
                
                x = data.to(device)
                x = torch.transpose(x, dim0=2, dim1=3)
                inputs = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
                outputs = self.net(inputs).squeeze().mean(dim=-1)
                batch_loss= torch.sum((outputs - self.c) ** 2, dim=1).cpu().numpy()
                loss.append(batch_loss)
        loss = np.concatenate(loss)

        auc_score, fps,tps = roc_auc_all(loss, delta_t, sigma)
        print("meann: {:.4f}, median: {:.4f}, auc:{:.4f}".format(np.mean(loss), np.median(loss),auc_score))# %%
        self.results['test_auc'] = auc_score
        return auc_score, fps,tps

    def pretrain(self, train_loader, args, device):
        """Pretrains the weights for the Deep SVDD network \phi via autoencoder."""

        self.ae_trainer = AETrainer(device=device)
        self.ae_net = self.ae_trainer.train(train_loader, self.ae_net, args)
        self.net = self.ae_net.encoder

    def save_model(self, export_model):
        """Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()

        torch.save({'c': self.c,
                    'net_dict': net_dict}, export_model)

    def load_model(self, model_path):
        """Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

# %%
import os
class DeepSADTrainer:

    def __init__(self, device: str = 'cuda'):

        self.device = device
       
        self.c = None


    def train(self, train_loader, net, args):
        self.args = args

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            print('Center c initialized.')

        # Training
        print('Starting training...')
        start_time = time.time()
        net.train()

        save_path = os.path.join(args.output_dir,args.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for epoch in range(args.n_epochs):



            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for data, semi_targets in train_loader:

                x = data.to(self.device)
                x = torch.transpose(x, dim0=2, dim1=3)
                inputs = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])

                semi_targets = semi_targets.to(self.device)
                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs).squeeze().mean(dim=-1)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                losses = torch.where(semi_targets == 0, dist, args.eta * ((dist + 1e-6) ** semi_targets.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1
            scheduler.step()
            # log epoch statistics

            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, args.n_epochs, epoch_train_time, loss_epoch / n_batches))
            torch.save({'c': self.c, 'net_dict': net.state_dict()}, os.path.join(save_path, "{}_{}.pt".format(args.name, epoch)))

        self.train_time = time.time() - start_time
        print('Training time: %.3f' % self.train_time)

        print('Finished training.')

        return net

    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = 0.0

        net.eval()
        with torch.no_grad():
            for data, _ in train_loader:
                # get the inputs of the batch
                x = data.to(self.device)
                x = torch.transpose(x, dim0=2, dim1=3)
                inputs = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
                outputs = net(inputs).squeeze() 
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c.mean(dim=-1)
class DeepSAD(object):

    def __init__(self, n_features, hidden_size, device):

        self.c = None  # hypersphere center c


        self.trainer = None

        self.ae_net = CNNAE(n_features, hidden_size).to(device)
        self.net = self.ae_net.encoder

        self.ae_trainer = None
        self.results = {
            'test_auc': None
        }

    def train(self, dataset, args, device: str = 'cuda'):

        self.trainer = DeepSADTrainer(device=device)
        # Get the model
        self.trainer.train(dataset, self.net, args)
        self.c = self.trainer.c

    def test(self, test_loader, delta_t, sigma, device):
        from utils import roc_auc_all
        import numpy as np
        self.net.eval()
        self.net.to(device)
        loss = []

        with torch.no_grad():
            for data in test_loader:
                
                x = data.to(device)
                x = torch.transpose(x, dim0=2, dim1=3)
                inputs = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
                outputs = self.net(inputs).squeeze().mean(dim=-1)
                batch_loss= torch.sum((outputs - self.c) ** 2, dim=1).cpu().numpy()
                loss.append(batch_loss)
        loss = np.concatenate(loss)

        auc_score, fps,tps = roc_auc_all(loss, delta_t, sigma)
        print("mean: {:.4f}, median: {:.4f}, auc:{:.4f}".format(np.mean(loss), np.median(loss),auc_score))# %%
        self.results['test_auc'] = auc_score
        return auc_score,fps,tps

    def pretrain(self, train_loader, args, device):

        self.ae_trainer = AETrainer(device=device)
        self.ae_net = self.ae_trainer.train(train_loader, self.ae_net, args)
        self.net = self.ae_net.encoder

    def save_model(self, export_model):
        """Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()

        torch.save({'c': self.c,
                    'net_dict': net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        """Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
# %%
