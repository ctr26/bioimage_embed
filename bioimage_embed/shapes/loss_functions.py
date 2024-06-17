from . import mds
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
import torch
import torch.nn.functional as F

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

    def loss(self, losses):
        losses = [getattr(self, loss)(self.D) for loss in losses]
        return torch.sum(torch.stack(losses))

    def normalize(self, D):
        # Frobenius norm
        return D / torch.norm(D, p="fro")

    def loss_mean(self, loss_function):
        # TODO fix this
        flat_D = self.D.view(-1, *self.D.size()[-2:])
        losses = []
        for i in range(flat_D.shape[0]):
            losses.append(loss_function(flat_D[i]))
        # losses = loss_function(flat_D)
        return torch.mean(torch.stack(losses))

    # def loss_mean(self,loss_function):
    #     batch_size, num_channels, height, width = self.D.shape
    #     flat_D = self.D.view(batch_size * num_channels, height, width)
    #     output = []
    #     for i in range(batch_size * num_channels):
    #         output.append(loss_function(flat_D[i]))
    #     return torch.mean(torch.stack(output))

    def diagonal_loss(self):
        return diagonal_loss(self.D)

    def symmetry_loss(self):
        return symmetry_loss(self.D)

    def non_negative_loss(self):
        return non_negative_loss(self.D)

    def triangle_inequality(self):
        return triangle_inequality_loss(self.D)

    def clockwise_order_loss(self):
        return clockwise_order_loss(self.D)

    def smoothness_loss(self):
        return smoothness_loss(self.D)


def triangle_inequality_loss_2D(distance_matrix):
    n = distance_matrix.shape[0]
    row_indices = torch.arange(n)
    combinations = torch.combinations(row_indices, 3)
    i, j, k = combinations.unbind(1)
    violation = distance_matrix[i, j] + distance_matrix[j, k] - distance_matrix[i, k]
    return torch.relu(violation).nanmean()


def triangle_inequality_loss(distance_matrix, n=1000):
    # Get the shape of the last two dimensions
    n, m = distance_matrix.shape[-2:]

    # Create a 2D grid of indices for each dimension
    i, j = torch.meshgrid(torch.arange(n), torch.arange(m))

    # Create a 1D array of indices for the last two dimensions
    indices = torch.arange(n * m)
    indices = indices[torch.randperm(indices.size()[0])]
    indices = indices[:n]
    # Get all combinations of three indices
    combinations = torch.combinations(indices, 3)

    # Convert 1D indices back to 2D
    i, j, k = combinations.unbind(1)
    i1, i2 = i // m, i % m
    j1, j2 = j // m, j % m
    k1, k2 = k // m, k % m

    # Calculate the violation of the triangle inequality
    violation = (
        distance_matrix[..., i1, i2]
        + distance_matrix[..., j1, j2]
        - distance_matrix[..., k1, k2]
    )

    # Return the mean of the rectified violation
    return torch.relu(violation).nanmean()


def clockwise_order_loss_2D(distance_matrix):
    n = distance_matrix.shape[0]
    row_indices = torch.arange(n)
    combinations = torch.combinations(row_indices, 2)
    i, j = combinations.unbind(1)

    expected_order = torch.arange(n).unsqueeze(0).repeat(n - 1, 1)
    expected_diff = (expected_order - expected_order.transpose(0, 1)) % n

    observed_diff = (distance_matrix[i, j] > distance_matrix[j, i]).long()
    loss = (observed_diff != expected_diff).float().mean()

    return loss


def clockwise_order_loss(distance_matrix):
    # Get the size of the last two dimensions
    n, m = distance_matrix.shape[-2:]

    # Create expected order tensor
    expected_order = torch.arange(n * m).reshape(1, 1, n, m).to(distance_matrix.device)

    # Create expected difference tensor
    expected_diff = (
        expected_order.unsqueeze(-1).unsqueeze(-1)
        - expected_order.unsqueeze(-3).unsqueeze(-3)
    ) % (n * m)

    # Expand dimensions for broadcasting
    distance_matrix_exp = distance_matrix.unsqueeze(-2).unsqueeze(-2)

    # Create observed difference tensor
    observed_diff = (distance_matrix_exp > distance_matrix_exp.transpose(-1, -2)).long()

    # Ensure expected_diff and observed_diff have the same shape
    expected_diff = expected_diff.expand_as(observed_diff)

    # Calculate loss
    loss = (observed_diff != expected_diff).float().mean()

    return loss


# def clockwise_order_loss(distance_matrix):
#     n = distance_matrix.shape[-1]
#     row_indices = torch.arange(n)
#     combinations = torch.combinations(row_indices, 2)
#     i, j = combinations.unbind(1)

#     expected_order = torch.arange(n).unsqueeze(0).repeat(n - 1, 1)
#     expected_diff = (expected_order - expected_order.transpose(0, 1)) % n

#     observed_diff = (distance_matrix[..., i, j] > distance_matrix[..., j, i]).long()
#     loss = (observed_diff != expected_diff).float().mean()

#     return loss


# def triangle_inequality_loss(distance_matrix):
#     # This function calculates the triangle inequality term
#     # for all combinations of rows in the distance matrix
#     n = distance_matrix.shape[-1]  # Assuming input shape is (batch_size, num_channels, n, n)
#     batch_size = distance_matrix.shape[0]
#     num_channels = distance_matrix.shape[1]

#     row_indices = torch.arange(n)
#     row_combinations = torch.combinations(row_indices, 3)  # All combinations of three rows
#     row1 = row_combinations[:, 0]
#     row2 = row_combinations[:, 1]
#     row3 = row_combinations[:, 2]

#     # Compute triangle_terms for each item in the batch and each color channel
#     triangle_terms = torch.sqrt(distance_matrix[:, :, row1, row2]) + torch.sqrt(distance_matrix[:, :, row1, row3]) - torch.sqrt(distance_matrix[:, :, row2, row3])
#     triangle_terms = torch.relu(triangle_terms)  # ReLU ensures we only get positive violations

#     # Average the loss across the batch size and the number of color channels
#     # TODO We're only worrying about nanmean here because sometimes the distance matrix is impossible and that will mean we get NaNs
#     return triangle_terms.nanmean()  # Return the average violation


def non_negative_loss(distance_matrix):
    # This function penalizes negative values in the distance matrix
    negative_values = torch.relu(
        -distance_matrix
    )  # ReLU gives us just the negative values
    return negative_values.mean()  # Return the average negative value


def diagonal_loss(distance_matrix):
    return F.mse_loss(
        torch.diagonal(distance_matrix),
        torch.zeros_like(torch.diagonal(distance_matrix)),
    )


def symmetry_loss(distance_matrix):
    return F.mse_loss(distance_matrix, distance_matrix.transpose(-2, -1))


# def symmetry_loss(distance_matrix):
#     return F.mse_loss(distance_matrix, torch.transpose(distance_matrix, dim0=-2, dim1=-1))


def has_self_intersections(coordinates):
    """
    This function checks if a given list of coordinates form a shape that has self-intersections.

    :param coordinates: List of tuples, each representing an (x, y) coordinate
    :return: Boolean value indicating whether the shape has self-intersections or not
    """
    line = LineString(coordinates)
    # Create a MultiPoint object from coordinates, this object has a buffer of 0
    # to ensure that every point in the MultiPoint has a minimum distance from the others
    m = MultiPoint([point for point in line.buffer(0).geoms])

    # If there are more points in m than in line, there is an intersection
    return len(m) > len(line.coords)


def coordinate_smoothness(coordinates):
    """
    This function computes the "smoothness" loss of a polygon defined by a list of coordinates.
    The smoothness loss is lower if the polygon's minimum angle is large.

    :param coordinates: Tensor of shape (N, 2) representing a list of N (x, y) coordinates
    :return: Scalar value representing the smoothness loss
    """
    # Shift the coordinates to get pairs of consecutive edges
    coordinates_shifted1 = torch.roll(coordinates, shifts=-1, dims=0)
    coordinates_shifted2 = torch.roll(coordinates, shifts=-2, dims=0)

    # Compute the vectors representing the edges
    vectors = coordinates_shifted1 - coordinates
    vectors_shifted = coordinates_shifted2 - coordinates_shifted1

    # Compute the dot product between each pair of consecutive vectors
    dot_product = (vectors * vectors_shifted).sum(dim=1)

    # Compute the magnitudes of the vectors
    magnitudes = torch.norm(vectors, dim=1) * torch.norm(vectors_shifted, dim=1)

    # Compute the cosine of the angle between each pair of consecutive edges
    cosine = dot_product / magnitudes

    # Compute the "smoothness" loss as the negative average cosine
    loss = -cosine.mean()

    return loss


def smoothness_loss(D):
    xy = mds.mds(D)
    return coordinate_smoothness(xy)
