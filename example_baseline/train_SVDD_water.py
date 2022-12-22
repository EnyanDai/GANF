
#%%
import os
import argparse
import torch
from models.RNN import RecurrentAE
import torch.nn.functional as F
from dataset import PMUTime
import numpy as np

parser = argparse.ArgumentParser()
# action

parser.add_argument('--data_dir', type=str, 
                    default='/data', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='/home/enyandai/code/checkpoint/model')
parser.add_argument('--dataset', type=str, default='C')
parser.add_argument('--model', type=str,
                    default='None')
parser.add_argument('--name',default='SVDD_Water_test')
parser.add_argument('--seed', type=int, default=11, help='Random seed to use.')
# made parameters
parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
# training params
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')


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
#%% load dataset
print("Loading dataset")
from dataset import load_water

train_loader, val_loader, test_loader, n_sensor = load_water("/home/enyandai/orginal_code/data/SWaT_Dataset_Attack_v0.csv", \
                                                                 args.batch_size)

# %%
from models.DeepSAD import DeepSVDD
model = DeepSVDD(n_sensor, args.hidden_size, device)
#%%
model.pretrain(train_loader, args, device)
model.train(train_loader, args, device)
#%%
save_path = os.path.join(args.output_dir,args.name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
model.save_model(os.path.join(save_path, "{}.pt".format(args.name)))
#%%
# for seed in range(10,21):
# model.load_model("/home/enyandai/orginal_code/checkpoint/model/SVDD_Water/SVDD_Water_39.pt")
model.net.eval()  
loss = []
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    for data in test_loader:
        
        x = data.to(device)
        x = torch.transpose(x, dim0=2, dim1=3)
        inputs = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        outputs = model.net(inputs).squeeze().mean(dim=-1)
        batch_loss= torch.sum((outputs - model.c) ** 2, dim=1).cpu().numpy()
        loss.append(batch_loss)
loss = np.concatenate(loss)
roc_test = roc_auc_score(np.asarray(test_loader.dataset.label.values,dtype=int),loss)
print("ROC: {:.4f}".format(roc_test))

# %%

# %%
