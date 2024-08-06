import numpy as np
import torch

def unbiased_HSIC(K, L):
    """Computes an unbiased estimator of HISC. This is equation (2) from the paper"""
    n = K.shape[0]
    ones = np.ones(shape=(n))
    np.fill_diagonal(K, val=0)
    np.fill_diagonal(L, val=0)
    
    trace = np.trace(np.dot(K, L))
    
    nominator1 = np.dot(np.dot(ones.T, K), ones)
    nominator2 = np.dot(np.dot(ones.T, L), ones)
    denominator = (n - 1) * (n - 2)
    middle = np.dot(nominator1, nominator2) / denominator
    
    multiplier1 = 2 / (n - 2)
    multiplier2 = np.dot(np.dot(ones.T, K), np.dot(L, ones))
    last = multiplier1 * multiplier2
    
    unbiased_hsic = 1 / (n * (n - 3)) * (trace + middle - last)
    return unbiased_hsic

def CKA(X, Y):
    """Computes the CKA of two matrices. This is equation (1) from the paper"""
    if isinstance(X, torch.Tensor):
        if X.device.type == "cuda" or Y.device.type == "cuda":
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
    
    nominator = unbiased_HSIC(np.dot(X, X.T), np.dot(Y, Y.T))
    denominator1 = unbiased_HSIC(np.dot(X, X.T), np.dot(X, X.T))
    denominator2 = unbiased_HSIC(np.dot(Y, Y.T), np.dot(Y, Y.T))
    
    cka = nominator / np.sqrt(denominator1 * denominator2)
    return cka