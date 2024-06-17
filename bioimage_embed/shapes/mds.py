import torch


def mds(d):
    """
    Multidimensional Scaling (MDS) in PyTorch.

    :param d: Distance matrix.
    :return: A matrix of x, y coordinates.
    """
    n = d.size(0)
    I = torch.eye(n)
    H = I - torch.ones((n, n)) / n

    S = -0.5 * H @ d @ H
    eigvals, eigvecs = S.symeig(eigenvectors=True)

    # Sort the eigenvalues and eigenvectors in decreasing order
    idx = eigvals.argsort(descending=True)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Take the square root of the two largest eigenvalues
    diag_lambdas = torch.diag(torch.sqrt(eigvals[:2]))

    return eigvecs[:, :2] @ diag_lambdas
