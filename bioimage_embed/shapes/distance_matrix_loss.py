from . import loss_functions as lf
import torch

LOSS_FUNCTIONS = {
    "diagonal_loss",
    "symmetry_loss",
    "non_negative_loss",
    "triangle_inequality_loss",
    "clockwise_order_loss",
}


class DistanceMatrixLoss:
    def __init__(self, D, norm=True) -> None:
        self.D = D
        self.norm = norm
        if norm:
            self.D = self.normalize(self.D)

    def loss():
        pass

    def normalize(self, D):
        # Frobenius norm
        return D / torch.norm(D, p="fro")

    def diagonal_loss(self):
        return lf.diagonal_loss(self.D)

    def symmetry_loss(self):
        return lf.symmetry_loss(self.D)

    def non_negative_loss(self):
        return lf.non_negative_loss(self.D)

    def triangle_inequality(self):
        return lf.triangle_inequality_loss(self.D)

    def clockwise_order_loss(self):
        return lf.clockwise_order_loss(self.D)
