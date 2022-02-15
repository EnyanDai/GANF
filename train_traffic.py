#%%
import os
import argparse
import torch
from models.GANF import GANF
import numpy as np


parser = argparse.ArgumentParser()
# files
parser.add_argument('--data_dir', type=str, 
                    default='./data', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/model')
parser.add_argument('--name',default='traffic')
parser.add_argument('--dataset', type=str, default='metr-la')
# restore
parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, default='None')
parser.add_argument('--seed', type=int, default=10, help='Random seed to use.')
# model parameters
parser.add_argument('--n_blocks', type=int, default=6, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--batch_norm', type=bool, default=False)
# training params
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--log_interval', type=int, default=5, help='How often to show loss statistics and save samples.')

parser.add_argument('--h_tol', type=float, default=1e-6)
parser.add_argument('--rho_max', type=float, default=1e16)
parser.add_argument('--max_iter', type=int, default=20)
parser.add_argument('--lambda1', type=float, default=0.0)
parser.add_argument('--rho_init', type=float, default=1.0)
parser.add_argument('--alpha_init', type=float, default=0.0)

args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


print(args)
import random
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#%%
print("Loading dataset")
from dataset import load_traffic

train_loader, val_loader, test_loader, n_sensor = load_traffic("{}/{}.h5".format(args.data_dir,args.dataset), \
                                                                args.batch_size)
#%%

rho = args.rho_init
alpha = args.alpha_init
lambda1 = args.lambda1
h_A_old = np.inf


max_iter = args.max_iter
rho_max = args.rho_max
h_tol = args.h_tol
epoch = 0

# initialize A
if args.graph != 'None':
    init = torch.load(args.graph).to(device).abs()
    print("Load graph from "+args.graph)
else:
    from torch.nn.init import xavier_uniform_
    init = torch.zeros([n_sensor, n_sensor])
    init = xavier_uniform_(init).abs()
    init = init.fill_diagonal_(0.0)
A = torch.tensor(init, requires_grad=True, device=device)

#%%
model = GANF(args.n_blocks, 1, args.hidden_size, args.n_hidden, dropout=0.0, batch_norm=args.batch_norm)
model = model.to(device)

if args.model != 'None':
    model.load_state_dict(torch.load(args.model))
    print('Load model from '+args.model)
#%%
from torch.nn.utils import clip_grad_value_
save_path = os.path.join(args.output_dir,args.name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
loss_best = 100
for _ in range(max_iter):

    while rho < rho_max:
        lr = args.lr #* np.math.pow(0.1, epoch // 100)
        optimizer = torch.optim.Adam([
            {'params':model.parameters(), 'weight_decay':args.weight_decay},
            {'params': [A]}], lr=lr, weight_decay=0.0)
        # train
        
        for _ in range(args.n_epochs):
            
            # train
            loss_train = []
            epoch += 1
            model.train()
            for x in train_loader:
                x = x.to(device)

                optimizer.zero_grad()
                A_hat = torch.divide(A.T,A.sum(dim=1).detach()).T
                loss = -model(x, A_hat)
                h = torch.trace(torch.matrix_exp(A_hat*A_hat)) - n_sensor
                total_loss = loss + 0.5 * rho * h * h + alpha * h

                total_loss.backward()
                clip_grad_value_(model.parameters(), 1)
                optimizer.step()
                loss_train.append(loss.item())
                A.data.copy_(torch.clamp(A.data, min=0, max=1))
            
            # evaluate 
            model.eval()
            loss_val = []
            with torch.no_grad():
                for x in val_loader:

                    x = x.to(device)
                    loss = -model(x,A_hat.data)
                    loss_val.append(loss.item())
            
            print('Epoch: {}, train -log_prob: {:.2f}, test -log_prob: {:.2f}, h: {}'\
                    .format(epoch, np.mean(loss_train), np.mean(loss_val), h.item()))

            if np.mean(loss_val) < loss_best:
                loss_best = np.mean(loss_val)
                print("save model {} epoch".format(epoch))
                torch.save(A.data,os.path.join(save_path, "graph_best.pt"))
                torch.save(model.state_dict(), os.path.join(save_path, "{}_best.pt".format(args.name)))
        
    
        print('rho: {}, alpha {}, h {}'.format(rho, alpha, h.item()))
        print('===========================================')
        torch.save(A.data,os.path.join(save_path, "graph_{}.pt".format(epoch)))
        torch.save(model.state_dict(), os.path.join(save_path, "{}_{}.pt".format(args.name, epoch)))
    
        del optimizer
        torch.cuda.empty_cache()
        
        if h.item() > 0.5 * h_A_old:
            rho *= 10
        else:
            break

    h_A_old = h.item()
    alpha += rho*h.item()

    if h_A_old <= h_tol or rho >=rho_max:
        break


# %%
lr = args.lr * 0.1
optimizer = torch.optim.Adam([
    {'params':model.parameters(), 'weight_decay':args.weight_decay},
    {'params': [A]}], lr=lr, weight_decay=0.0)
# train

for _ in range(100):
    loss_train = []
    epoch += 1
    model.train()
    for x in train_loader:
        x = x.to(device)

        optimizer.zero_grad()
        A_hat = torch.divide(A.T,A.sum(dim=1).detach()).T
        loss = -model(x, A_hat)
        h = torch.trace(torch.matrix_exp(A_hat*A_hat)) - n_sensor
        total_loss = loss + 0.5 * rho * h * h + alpha * h 

        total_loss.backward()
        clip_grad_value_(model.parameters(), 1)
        optimizer.step()
        loss_train.append(loss.item())
        A.data.copy_(torch.clamp(A.data, min=0, max=1))

    model.eval()
    loss_val = []
    print(A.max())
    with torch.no_grad():
        for x in val_loader:

            x = x.to(device)
            loss = -model(x,A_hat.data)
            loss_val.append(loss.item())
    
    print('Epoch: {}, train -log_prob: {:.2f}, test -log_prob: {:.2f}, h: {}'\
            .format(epoch, np.mean(loss_train), np.mean(loss_val), h.item()))

    if np.mean(loss_val) < loss_best:
        loss_best = np.mean(loss_val)
        print("save model {} epoch".format(epoch))
        torch.save(A.data,os.path.join(save_path, "graph_best.pt"))
        torch.save(model.state_dict(), os.path.join(save_path, "{}_best.pt".format(args.name)))

    if epoch % args.log_interval==0:
        torch.save(A.data,os.path.join(save_path, "graph_{}.pt".format(epoch)))
        torch.save(model.state_dict(), os.path.join(save_path, "{}_{}.pt".format(args.name, epoch)))

#%%
