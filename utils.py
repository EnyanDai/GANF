#%%
import torch

def h(A):
    return torch.trace(torch.matrix_exp(A*A)) - A.shape[0]

def normalize(A):
    D = A.sum(dim=0)
    D_inv = D.pow_(-1)
    D_inv.masked_fill_(D_inv == float('inf'), 0)
    
    return A * D_inv
    
def thresholding(A, thre):
    return torch.where(A.abs()>thre, A, torch.scalar_tensor(0.0, dtype=torch.float32, device=A.device))

def binarize(A, thre):
    return torch.where(A.abs()>thre, 1.0, 0.0)
# %%
import pandas as pd
def get_timestamp(stamps):
    return (stamps - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
# %%
import numpy as np
from sklearn.metrics import auc
def roc_auc(label_time, pred, negative_sample, sigma):
    negative_sample = np.sort(negative_sample)[::-1]
    thresholds = list(negative_sample[::int(len(negative_sample)/50)])
    thresholds.append(negative_sample[-1])
    tps=[]
    fps=[]

    for thre in thresholds:
        pred_pos = pred[pred>thre]

        tp = 0
        for i in range(len(label_time)):
            start_time = label_time[i] - pd.Timedelta(30, unit='min')
            end_time = label_time[i] + pd.Timedelta(30, unit='min')

            detected_event = pred_pos[str(start_time): str(end_time)]
            if len(detected_event)>0:
                timestamps = get_timestamp(detected_event.index)
                delta_t = np.min(np.abs(timestamps.values - get_timestamp(label_time[i])))
                tp += np.exp(-np.power(delta_t/sigma,2))
        tp = tp/len(label_time)
        tps.append(tp)

        fp = (negative_sample>thre).sum()/len(negative_sample)
        fps.append(fp)
    return auc(fps,tps), (fps,tps)
# %%
def roc_auc_all(loss_np, delta_t, sigma):

    ground_truth = np.exp(-np.power((delta_t.values)/sigma,2))

    loss_sort = np.sort(loss_np)[::-1]
    thresholds = list(loss_sort[::int(len(loss_sort)/50)])
    thresholds.append(loss_sort[-1])

    n_pos = ground_truth.sum()
    n_neg = (1-ground_truth).sum()
    tps = []
    fps = []
    for thre in thresholds:
        pred_pos = loss_np>thre

        tp = ground_truth[pred_pos].sum()/n_pos
        fp = (1-ground_truth[pred_pos]).sum()/n_neg
        tps.append(tp)
        fps.append(fp)

    auc_score = auc(fps, tps)
    return auc_score, fps, tps