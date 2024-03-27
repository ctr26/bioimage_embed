# Imports when necessary
import numpy as np
import torch
import logging
import sklearn
import skimage as sk
import scipy.spatial
from scipy.interpolate import splprep, splev
import matplotlib.image
import matplotlib.pyplot as plt
import glob

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS
 
from bioimage_embed import shapes
import bioimage_embed
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from bioimage_embed.lightning import DataModule

from torchvision import datasets

from bioimage_embed.shapes.mds import mds

from bioimage_embed.shapes.transforms import (
    CropCentroidPipeline,
    CoordsToDistogram,
    ImageToCoords,
    RotateIndexingClockwise,
)

logger = logging.getLogger(__name__)

# Where is the datat I want to transform
dataset = f"/nfs/research/uhlmann/afoix/bbbc010/BBBC010_v1_foreground_eachworm/"

##########################################################################
####### Simplified version in order to make the things properly work #####
##########################################################################

def find_contour(mask):
    contour = sk.measure.find_contours(mask, 0.8)[0]
    x, y = contour[:, 0], contour[:, 1]
    return x, y

def spline_interpolation(x, y):
    sparsity_contour = 4 # Sparsity of the contour. Dropping some of the sample (points) to make the spline smoother
    tck, u = splprep([x[::sparsity_contour], y[::sparsity_contour]], s = 0)
    sample_points = 200
    # How many times to sample the spline
    new_u = np.linspace(u.min(), u.max(), sample_points) # Last parameter is how dense is our spline, how many points. 
    # Evaluate the spline
    x_spline, y_spline = splev(new_u, tck)
    return x_spline, y_spline

def build_distance_matrix(x_reinterpolated, y_reinterpolated):
    reinterpolated_contour = np.column_stack([x_reinterpolated, y_reinterpolated])
    dm = scipy.spatial.distance_matrix(reinterpolated_contour, reinterpolated_contour)
    return dm

def dist_to_coords(dst_mat):
  embedding = MDS(n_components=2, dissimilarity='precomputed')
  return embedding.fit_transform(dst_mat)
    

# Simplified version for test
def process_png_file(mask_path):
    # Perform specific action for each PNG file
    print("Processing:", mask_path)
    mask = plt.imread(mask_path)
    
    # Get the contour
    x, y = find_contour(mask)

    # Reinterpolate (spline)
    x_reinterpolated, y_reinterpolated = spline_interpolation(x, y)
    plt.scatter(x_reinterpolated, y_reinterpolated, s=6)
    plt.savefig(f'results/reconstruction/original_contour{i}.png')
    plt.clf()

    # Build the distance matrix
    dm = build_distance_matrix(x_reinterpolated, y_reinterpolated)
    # print("Distance matrix")
    # print(dm)

    # Reconstruction coordinates and matrix (MDS)
    reconstructed_coords = dist_to_coords(dm)
    print(reconstructed_coords)
    plt.scatter(*zip(*reconstructed_coords), s=6)
    plt.savefig(f'results/reconstruction/reconstructed_contour{i}.png')
    plt.clf()
    reconstructed_matrix = euclidean_distances(reconstructed_coords)

    # Error with matrix
    err = np.average(dm - reconstructed_matrix)
    print(f"Dist error is: {err}")

# Specify the folder path containing PNG files
folder_path = "/nfs/research/uhlmann/afoix/bbbc010/BBBC010_v1_foreground_eachworm/*/*.png"

# Use glob to find all PNG files in the folder
png_files = glob.glob(folder_path)

# Iterate through all PNG files found
for i, file_path in enumerate(png_files):
    # Process the PNG file
    process_png_file(file_path)




########################################
############# Other code ###############
########################################

# # Needed variables 
# window_size = 256 # needs to be the same as the latent space size
# interp_size = 256 # latent space size needs to match the window size

# # This crops the image using the centroid by window sizes. (remember to removed and see what happens)
# transform_crop = CropCentroidPipeline(window_size) 

# # From the coordinates of the distance matrix, this is actually building the distance matrix
# transform_coord_to_dist = CoordsToDistogram(interp_size, matrix_normalised=False)

# # It takes the images and converts it into a numpy array  of the image and the size
# transform_coords = ImageToCoords(window_size)

# # Combination of transforms
# transform_mask_to_gray = transforms.Compose([transforms.Grayscale(1)])

# transform_mask_to_crop = transforms.Compose(
#         [
#             # transforms.ToTensor(),
#             transform_mask_to_gray,
#             transform_crop,
#         ]
#     )

# transform_mask_to_coords = transforms.Compose(
#         [
#             transform_mask_to_crop,
#             transform_coords,
#         ]
#     )

# transform_mask_to_dist = transforms.Compose(
#         [
#             transform_mask_to_coords,
#             transform_coord_to_dist,
#         ]
#     )

# def dist_to_coords(dst_mat):
#   embedding = MDS(n_components=2, dissimilarity='precomputed', max_iter=1)
#   return embedding.fit_transform(dst_mat)
  
  #coords_prime = MDS(
    #n_components=2, dissimilarity="precomputed", random_state=0).fit_transform(dst_mat)
  
  #return coords_prime
  #return mds(dst_mat)
  
  # from https://math.stackexchange.com/a/423898 and https://stackoverflow.com/a/17177833/16632916
#   m = np.zeros(shape=dst_mat.shape)
#   for i in range(dst_mat.shape[0]):
#     for j in range(dst_mat.shape[1]):
#       m[i,j]= 0.5*(dst_mat[0, j]**2 + dst_mat[i, 0]**2 - dst_mat[i, j]**2)
#   eigenvalues, eigenvectors = np.linalg.eig(m)
#   print(f'm:{m}')
#   print(f'eigenvalues:{eigenvalues}')
#   print(f'eigenvectors:{eigenvectors}')
#   return np.sqrt(eigenvalues)*eigenvectors

# # Convert your image to gray scale
# gray2rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1))

# # choose the transformation you want to apply to your data and Compose
# transform = transforms.Compose(
#         [
#             transform_mask_to_dist,
#             transforms.ToTensor(),
#             RotateIndexingClockwise(p=1), # This module effectively allows for random clockwise rotations of input images with a specified probability.
#             gray2rgb,
#         ]
#     )

# transforms_dict = {
#         "none": transform_mask_to_gray,
#         "transform_crop": transform_mask_to_crop,
#         "transform_dist": transform_mask_to_dist,
#         "transform_coords": transform_mask_to_coords,
#     }



# diagonal = np.diag(dm)

# if np.all(diagonal == 0):
#   print("All elements in the diagonal are zeros.")
#   dataset_raw[i][0].save(f'original_{i}.png')
#   np.save(f"random_matrix_{i}.npy", dataset_trans[i][0][0])
#   matplotlib.image.imsave(f'dist_mat_{i}.png', dataset_trans[i][0][0])
#   coords = dist_to_coords(dataset_trans[i][0][0])
#   print(coords)
#   x, y = list(zip(*coords))
#   plt.scatter(x_reinterpolated, y_reinterpolated)
#   plt.savefig(f'mask_{i}.png')
#   plt.clf()
#   fig, ax = plt.subplots(1, 4, figsize=(20, 5))
#   ax[0].imshow(mask)
#   ax[1].scatter(x_reinterpolated, y_reinterpolated)
#   ax[1].imshow(dm)
#   ax[3].scatter(x, y)
#   fig.savefig(f'combined_{i}.png')
# else:
#   print("Not all elements in the diagonal are zeros.")



# # Apply transform to find which images don't work
# dataset_raw = datasets.ImageFolder(dataset)
# dataset_contours = datasets.ImageFolder(dataset, transform=transform_mask_to_coords)
# dataset_trans = datasets.ImageFolder(dataset, transform=transform)

# # This is a single image distance matrix
# for i in range(0, 10):
#     print(dataset_trans[i][0][0])
#     diagonal = np.diag(dataset_trans[i][0][0])
#     if np.all(diagonal == 0):
#         print("All elements in the diagonal are zeros.")
#         dataset_raw[i][0].save(f'original_{i}.png')
#         np.save(f"random_matrix_{i}.npy", dataset_trans[i][0][0])
#         matplotlib.image.imsave(f'dist_mat_{i}.png', dataset_trans[i][0][0])
#         coords = dist_to_coords(dataset_trans[i][0][0])
#         print(coords)
#         x, y = list(zip(*coords))
#         plt.scatter(x, y)
#         plt.savefig(f'mask_{i}.png')
#         plt.clf()
#         fig, ax = plt.subplots(1, 4, figsize=(20, 5))
#         ax[0].imshow(dataset_raw[i][0])
#         ax[1].imshow(dataset_trans[i][0][0])
#         ax[2].scatter(dataset_contours[i][0][0], dataset_contours[i][0][1])
#         ax[3].scatter(x, y)
#         fig.savefig(f'combined_{i}.png')
#     else:
#         print("Not all elements in the diagonal are zeros.")
