import numpy as np

#Building a SVD Function from scratch
def compute_svd(A,full_matrices = True):
    ATA = A.T @ A
    eigvals, V = np.linalg.eigh(ATA)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]
    S = np.sqrt(eigvals)
    U = A @ V /S
    

    return U,S, V.T

    