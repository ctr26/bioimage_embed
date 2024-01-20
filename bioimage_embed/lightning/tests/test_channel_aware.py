import pytest
import torch

from .. torch import generalised_js_loss_z_loss, generalised_euclidean_distance_z_loss
# Fixture for batch sizes
@pytest.fixture(params=[1, 16])
def batch_size(request):
    return request.param

# Fixture for channel sizes
@pytest.fixture(params=[1, 3, 5])
def channel_size(request):
    return request.param

# Fixture for z dimensions
@pytest.fixture(params=[2**i for i in range(2, 9)])
def z_dim(request):
    return request.param

# Fixture for generating tensors
@pytest.fixture
def input_tensor(batch_size, channel_size, z_dim):
    return torch.randn(batch_size, channel_size, z_dim)

# Test for generalised_js_loss_z_loss
def test_generalised_js_loss_z_loss(input_tensor):
    # Call the function and make assertions
    loss = generalised_js_loss_z_loss(input_tensor)
    assert loss is not None  # Example assertion

# Test for generalised_euclidean_distance_z_loss
def test_generalised_euclidean_distance_z_loss(input_tensor):
    # Call the function and make assertions
    loss = generalised_euclidean_distance_z_loss(input_tensor)
    assert loss is not None  # Example assertion