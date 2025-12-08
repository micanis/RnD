# src/ctr-gcn/data/preprocess.py
import numpy as np
from config import INPUT_T

def temporal_resample(frames, T=INPUT_T):
    src_T = len(frames)
    idx = np.linspace(0, max(0, src_T - 1), T).astype(int)
    return np.stack([frames[i] for i in idx], axis=0)

def center_scale(x):  # x: (C,T,V,M)
    ref = x[:, :, :1, :]
    x = x - ref
    scale = np.linalg.norm(x.reshape(3, -1), axis=0).mean() + 1e-6
    return x / scale

def to_tensor(frames):
    arr = temporal_resample(frames)        # (T,M,V,3)
    arr = np.transpose(arr, (3, 0, 2, 1))  # (C,T,V,M)
    arr = center_scale(arr)
    return arr.astype(np.float32)

def bones(arr, edges):
    C,T,V,M = arr.shape
    b = np.zeros_like(arr)
    for (u,v) in edges:
        b[:,:,u,:] = arr[:,:,u,:] - arr[:,:,v,:]
    return b

def motion(arr):
    m = np.zeros_like(arr)
    m[:,1:,:,:] = arr[:,1:,:,:] - arr[:,:-1,:,:]
    return m
