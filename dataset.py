#%%
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

# %%
from torch.utils.data import DataLoader
def load_traffic(root, batch_size):
    """
    Load traffic dataset
    return train_loader, val_loader, test_loader
    """
    df = pd.read_hdf(root)
    df = df.reset_index()
    df = df.rename(columns={"index":"utc"})
    df["utc"] = pd.to_datetime(df["utc"], unit="s")
    df = df.set_index("utc")
    n_sensor = len(df.columns)

    mean = df.values.flatten().mean()
    std = df.values.flatten().std()

    df = (df - mean)/std
    df = df.sort_index()
    # split the dataset
    train_df = df.iloc[:int(0.75*len(df))]
    val_df = df.iloc[int(0.75*len(df)):int(0.875*len(df))]
    test_df = df.iloc[int(0.75*len(df)):]

    train_loader = DataLoader(Traffic(train_df), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Traffic(val_df), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Traffic(test_df), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor  

class Traffic(Dataset):
    def __init__(self, df, window_size=12, stride_size=1):
        super(Traffic, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.time = self.preprocess(df)
    
    def preprocess(self, df):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(5*self.window_size,unit='min')

        return df.values, start_idx[idx_mask], df.index[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1)

def load_water(root, batch_size,label=False):
    
    data = pd.read_csv(root)
    data = data.rename(columns={"Normal/Attack":"label"})
    data.label[data.label!="Normal"]=1
    data.label[data.label=="Normal"]=0
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data = data.set_index("Timestamp")

    #%%
    feature = data.iloc[:,:51]
    mean_df = feature.mean(axis=0)
    std_df = feature.std(axis=0)

    norm_feature = (feature-mean_df)/std_df
    norm_feature = norm_feature.dropna(axis=1)
    n_sensor = len(norm_feature.columns)

    train_df = norm_feature.iloc[:int(0.6*len(data))]
    train_label = data.label.iloc[:int(0.6*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = data.label.iloc[int(0.6*len(data)):int(0.8*len(data))]
    
    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = data.label.iloc[int(0.8*len(data)):]
    if label:
        train_loader = DataLoader(WaterLabel(train_df,train_label), batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(Water(train_df,train_label), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Water(val_df,val_label), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Water(test_df,test_label), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor

class Water(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10):
        super(Water, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        return df.values, start_idx[idx_mask], label[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1)


class WaterLabel(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10):
        super(WaterLabel, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        self.label = 1.0-2*self.label 
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        return df.values, start_idx[idx_mask], label[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1),self.label[index]