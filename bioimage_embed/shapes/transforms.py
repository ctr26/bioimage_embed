import numpy as np
from skimage.draw import polygon2mask

# Note - you must have torchvision installed for this example
from torchvision import transforms
from skimage.measure import regionprops
from torchvision.transforms.functional import crop
import torch
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from skimage.measure import find_contours
from torch import nn

from . import contours


class cropCentroid(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        return self.crop_centroid(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"

    def crop_centroid(self, image, size):
        # TODO: Only works for grayscale images
        np_image = np.array(image)
        im_height, im_width = np_image.shape
        height, width = size, size

        properties = regionprops(np_image.astype(int), np_image.astype(int))
        center_of_mass = properties[0].centroid
        # weighted_center_of_mass = properties[0].weighted_centroid
        top = int(center_of_mass[0] - size / 2)
        bottom = top + height
        left = int(center_of_mass[1] - size / 2)
        right = left + width

        # if left <= 0 or top <= 0 or right >= im_height or bottom >= im_width:
        # return None
        # return Image.eval(crop(image,top,left,height,width), (lambda x: 0))
        # TODO find bad croppings
        # if ((top <= 0)  or (top+height >= im_height)  or (left <= 0) or (left+width >= 0) ):
        # return Image.eval(crop(image,top,left,height,width), (lambda x: 0))
        return crop(image, top, left, height, width)


class DistogramToCoords(torch.nn.Module):
    def __init__(self, size=256 + 128):
        super().__init__()
        self.size = size

    def forward(self, image):
        # return(self.get_points_from_dist_C(image,self.size))
        return self.get_points_from_dist_BC(image, self.size)

    def __repr__(self):
        return self.__class__.__name__

    def get_points_from_dist(self, image, method="MDS"):
        if method == "MDS":
            return self.get_points_from_dist_MDS(image)
        if method == "Matrix":
            return self.calculate_positions(image)

    def get_points_from_dist_MDS(self, image):
        return MDS(
            n_components=2, dissimilarity="precomputed", random_state=0
        ).fit_transform(image)

    def get_points_from_dist_vec(self):
        return np.vectorize(self.get_points_from_dist)

    def get_points_from_dist_BC(self, image, size):
        flat = np.reshape(image, (-1, image.shape[-2], image.shape[-1]))
        coords = np.stack([self.get_points_from_dist(arr) for arr in flat]).reshape(
            *image.shape[-4:-1], -1
        )
        coords_scaled = (coords * size) + (size / 2)  # TODO Check this scaling
        return coords_scaled

    def x_coord_of_point(self, D, j):
        return (D[0, j] ** 2 + D[0, 1] ** 2 - D[1, j] ** 2) / (2 * D[0, 1])

    def coords_of_point(self, D, j):
        x = self.x_coord_of_point(D, j)
        return np.array([x, np.sqrt(D[0, j] ** 2 - x**2)])

    def calculate_positions(self, D):
        (m, n) = D.shape
        P = np.zeros((n, 2))
        tr = (min(min(D[2, 0:2]), min(D[2, 3:n])) / 2) ** 2
        P[1, 0] = D[0, 1]
        P[2, :] = self.coords_of_point(D, 2)
        for j in range(3, n):
            P[j, :] = self.coords_of_point(D, j)
            if abs(np.dot(P[j, :] - P[2, :], P[j, :] - P[2, :]) - D[2, j] ** 2) > tr:
                P[j, 1] = -P[j, 1]
        return P

    #     # for i in flat:

    #     # flat = torch.flatten(torch.tensor(image), start_dim=0, end_dim=1)
    #     # np.vectorize(self.get_points_from_dist)(np.array(flat))
    #     # self.get_points_from_dist_vec(flat)
    #     # # coords = np.apply_along_axis(self.get_points_from_dist_vec, axis=0, np.array(flat))
    #     # coords = self.get_points_from_dist(np.vectorize(flat)).reshape(image.shape)
    #     # # coords = np.apply_over_axes(self.get_points_from_dist, image, [2,3])
    #     # coords_scaled = (coords*size)+(size/2)  # TODO Check this scaling
    #     # return coords_scaled

    def get_points_from_dist_C(self, tensor, size):
        dist_list = []
        np_tensor = np.array(tensor)
        for i in range(np_tensor.shape[0]):
            image = np_tensor[0, :, :]
            coords = self.get_points_from_dist(image.squeeze())
            coords_scaled = (coords * size) + (size / 2)  # TODO Check this scaling
            dist_list.append(coords_scaled)
        return torch.tensor(np.array(dist_list))


class ImageToDistogram(torch.nn.Module):
    def __init__(self, size, matrix_normalised=False):
        super().__init__()
        self.size = size
        self.matrix_normalised = matrix_normalised

    def forward(self, img):
        # return self.get_distogram(img, self.size)
        return self.pipeline()(img)

    def pipeline(self):
        components = [
            ImageToCoords(self.size),
            CoordsToDistogram(self.size, matrix_normalised=self.matrix_normalised),
        ]
        return transforms.Compose(components)


class CoordsToDistogram(torch.nn.Module):
    def __init__(self, size, matrix_normalised=False):
        super().__init__()
        self.size = size
        self.matrix_normalised = matrix_normalised

    def forward(self, coords):
        return self.get_distogram(coords,
                                matrix_normalised=self.matrix_normalised)

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"

    def get_distogram(self, coords, matrix_normalised=False):

        xii, yii = coords
        distance_matrix = euclidean_distances(np.array([xii, yii]).T) / (
            np.sqrt(2) * self.size
        )
        # TODO size should be shape of matrix and the normalisation should be
        # D / (np.linalg.norm(x.shape[-2:]))

        norm = np.linalg.norm(distance_matrix, "fro")
        if matrix_normalised:
            return distance_matrix / np.linalg.norm(distance_matrix, "fro")
        if not matrix_normalised:
            return distance_matrix / np.linalg.norm([self.size, self.size])


def find_longest_array(arrays):
    lengths = [len(arr.flatten()) for arr in arrays]
    max_length_index = np.argmax(lengths)
    return arrays[max_length_index]


class ImageToCoords(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        # return self.get_distogram(img, self.size)
        return self.get_coords_pil(
            img,
            self.size,
        )

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"

    def get_coords_pil(self, image, size):
        return self.get_coords(np.array(image), size)

    def get_coords_C(
        self,
        tensor,
        size,
    ):
        coords_list = []
        np_tensor = np.array(tensor)
        for i in range(np_tensor.shape[0]):
            image = np_tensor[0, :, :]
            coords_list.append(self.get_coords(image, size))
        return torch.tensor(np.array(coords_list))

    def get_coords(self, image, size, method="uniform_spline", contour_level=0.8):
        contour_list = find_contours(np.array(image), contour_level)
        contour = find_longest_array(contour_list)
        if method == "uniform_spline":
            return contours.uniform_spline_resample_contour(contour=contour, size=size)
        if method == "cubic_polar":
            return contours.cubic_polar_resample_contour(contour=contour, size=size)


class VerticesToMask(torch.nn.Module):
    # https://pypi.org/project/bentley-ottmann/
    # Should check that the shape is a "simple" polygon
    # Options if it isn't, error or try to reorder the coords
    # For instance, find https://en.wikipedia.org/wiki/Hamiltonian_path
    # Basically the travelling salesman problem though

    # Alternative: is to enforce simple polygonality in loss function,
    # Don't know how though
    def __init__(self, size=256 + 128):
        super().__init__()
        self.size = size

    def forward(self, x):
        # return self.vertices_to_mask(x, mask_shape=(self.size, self.size))
        return self.vertices_to_mask_BC(x, mask_shape=(self.size, self.size))

    def vertices_to_mask(self, vertices, mask_shape=(128, 128)):
        mask_list = []
        for channel in vertices:
            # channel_scaled = (channel + mask_shape[0]/2) * mask_shape[0]/2
            mask_list.append(polygon2mask(mask_shape, channel))
        return torch.tensor(np.array(mask_list))

    def vertices_to_mask_BC(self, vertices, mask_shape=(128, 128)):
        flat = np.reshape(vertices, (-1, vertices.shape[-2], vertices.shape[-1]))
        masks = np.stack([polygon2mask(mask_shape, arr) for arr in flat]).reshape(
            *vertices.shape[-4:-2], *mask_shape
        )
        # shape = masks.shape
        return masks


class CropCentroidPipeline(torch.nn.Module):
    def __init__(self, window_size, num_output_channels=1):
        super().__init__()
        self.window_size = window_size
        self.pipeline = transforms.Compose(
            [
                # transforms.ToPILImage(),
                cropCentroid(self.window_size),
                transforms.ToTensor(),
                # transforms.Normalize(0, 1),
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=num_output_channels),
                # transforms.ToTensor()
                # transforms.RandomCrop((512, 512)),
                # transforms.ConvertImageDtype(torch.bool)
            ]
        )

    def forward(self, x):
        # try:
        return self.pipeline(x)
        # except:
        # return None

        
class MaskToDistogramPipeline(torch.nn.Module):
    def __init__(self, window_size, interp_size=128, matrix_normalised=False):
        super().__init__()
        self.window_size = window_size
        self.interp_size = interp_size
        self.pipeline = transforms.Compose(
            [
                CropCentroidPipeline(self.window_size),
                ImageToDistogram(self.interp_size, matrix_normalised=matrix_normalised),
            ]
        )

    def forward(self, x):
        # try:
        return self.pipeline(x)
        # except:
        # return None


class DistogramToMaskPipeline(torch.nn.Module):
    """
    Placeholder class
    """

    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.pipeline = transforms.Compose(
            [DistogramToCoords(self.window_size), VerticesToMask(self.window_size)]
        )

    def forward(self, x):
        # try:
        return self.pipeline(x)
        # except:
        # return None


class AsymmetricDistogramToMaskPipeline(torch.nn.Module):
    """
    Placeholder class
    """

    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.pipeline = transforms.Compose(
            [
                AsymmetricDistogramToSymmetricDistogram(),
                DistogramToMaskPipeline(self.window_size),
            ]
        )

    def forward(self, x):
        # try:
        #     return self.pipeline(x)
        # except:
        #     return None
        return self.pipeline(x)


class AsymmetricDistogramToCoordsPipeline(torch.nn.Module):
    """
    Placeholder class
    """

    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.pipeline = transforms.Compose(
            [
                AsymmetricDistogramToSymmetricDistogram(),
                DistogramToCoords(self.window_size),
            ]
        )

    def forward(self, x):
        # try:
        #     return self.pipeline(x)
        # except:
        #     return None
        return self.pipeline(x)


class AsymmetricDistogramToSymmetricDistogram(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.asym_dist_to_sym_dist(x)

    def asym_dist_to_sym_dist(self, asymm_dist):
        dist_stack = np.stack([asymm_dist, asymm_dist.transpose(0, 1, 3, 2)], axis=0)

        sym_dist = np.max(dist_stack, axis=0)
        return torch.tensor(np.array(sym_dist))


class RotateIndexingClockwise(nn.Module):
    def __init__(self, max_rotations=None, p=1.0):
        super(RotateIndexingClockwise, self).__init__()
        self.max_rotations = max_rotations
        self.probability = p

    def forward(self, img):
        if np.random.rand() < self.probability:
            if self.max_rotations is None:
                self.max_rotations = img.shape[0]
            num_rotations = np.random.randint(0, self.max_rotations)
            img = np.roll(
                img.numpy(), shift=[num_rotations, num_rotations], axis=[0, 1]
            )
        return torch.from_numpy(img)
