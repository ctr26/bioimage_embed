import numpy as np
import imageio.v3 as iio
import skimage as sk
from scipy.interpolate import splprep, splev
import scipy.spatial
import argparse
import pathlib
import types
import glob
import os

# misc helpers
###############################################################################

def vprint(tgtlvl, msg, pfx = f"{'':<5}"):
  try:
    if (tgtlvl <= vprint.lvl):
      print(f"{pfx}{msg}")
  except AttributeError:
    print("verbosity level not set, defaulting to 0")
    vprint.lvl = 0
    vprint(tgtlvl, msg)

def rgb2grey(rgb, cr = 0.2989, cg = 0.5870, cb = 0.1140):
  """Turn an rgb array into a greyscale array using the following reduction:
     grey = cr * r + cg * g + cb * b

    :param rgb: The rgb array
    :param cr: The red coefficient
    :param cg: The green coefficient
    :param cb: The blue coefficient

    :returns: The greyscale array.
    """
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  return cr * r + cg * g + cb * b

##########################################################################
####### Simplified version in order to make the things properly work #####
##########################################################################

def find_longest_contour(mask):
    if len(mask.shape) == 3: # (lines, columns, number of channels)
      mask = rgb2grey(mask)
    contours = sk.measure.find_contours(mask, 0.8)
    vprint(4, f'len(contours) {len(contours)}')
    contours = sorted(contours, key=lambda x: len(x), reverse=True)
    x, y = contours[0][:, 0], contours[0][:, 1]
    return x, y

def spline_interpolation(x, y, raw_sampling_sparsity, spline_sampling):
    # Sparsity of the contour. Dropping some of the sample (points) to make the spline smoother
    raw_sampling_sparsity = max(1, raw_sampling_sparsity)
    vprint(3, f'running with raw_sampling_sparsity {raw_sampling_sparsity} and spline_sampling {spline_sampling}')
    vprint(3, f'x.shape {x.shape} y.shape {y.shape}')
    tck, u = splprep([x[::raw_sampling_sparsity], y[::raw_sampling_sparsity]], s = 0, per = True)
    # How many times to sample the spline
    new_u = np.linspace(u.min(), u.max(), spline_sampling) # Last parameter is how dense is our spline, how many points.
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

def mask2distmatrix(mask, raw_sampling_sparsity=1, spline_sampling=512):
  vprint(3, f'running with raw_sampling_sparsity {raw_sampling_sparsity} and spline_sampling {spline_sampling}')
  # extract mask contour
  x, y = find_longest_contour(mask)
  vprint(3, f'found contour shape x {x.shape} y {y.shape}')
  # Reinterpolate (spline)
  x_reinterpolated, y_reinterpolated = spline_interpolation(x, y, raw_sampling_sparsity, spline_sampling)
  # Build the distance matrix
  dm = build_distance_matrix(x_reinterpolated, y_reinterpolated)
  vprint(3, f'created distance matrix shape {dm.shape}')
  return dm

def masks2distmatrices(params):

  vprint(1, 'loading base dataset')

  if not params.mask_dataset_path:
    sys.exit("no mask dataset provided")
  if not params.output_path:
    p = pathlib.Path(params.mask_dataset_path)
    params.output_path=p.joinpath(p.parent, p.name+'_distmat')

  vprint(2, f'>>>> params.mask_dataset_path: {params.mask_dataset_path}')
  vprint(2, f'>>>> params.mask_dataset_path: {next(os.walk(params.mask_dataset_path))[1]}')
  vprint(2, f'>>>> params.output_path: {params.output_path}')
  pathlib.Path(params.output_path).mkdir(parents=True, exist_ok=True)
  class_folders = next(os.walk(params.mask_dataset_path))[1]
  vprint(2, f'>>>> class_folders: {class_folders}')
  for class_folder in class_folders:
    vprint(2, f'>>>> class_folder: {class_folder}')
    output_class_folder=os.path.join(params.output_path, class_folder)
    vprint(2, f'creating output class folder: {output_class_folder}')
    pathlib.Path(output_class_folder).mkdir(parents=True, exist_ok=True)
    for mask_png in glob.glob(params.mask_dataset_path+'/'+class_folder+'/'+'*.png'):
      vprint(3, f'{"-"*80}')
      vprint(3, f'working on {mask_png}')
      filename = os.path.basename(mask_png).split('.')[0]
      vprint(3, f'filename {filename}')
      mask = iio.imread(mask_png)
      dm = mask2distmatrix(mask, params.raw_sampling_sparsity, params.spline_sampling)
      output_file_name=f"{output_class_folder}/{filename}.npy"
      vprint(3, f'saving {output_file_name}')
      vprint(3, f'{"-"*80}')
      np.save(output_file_name, dm)


  #print('loading base dataset')
  #dataset = datasets.ImageFolder(mask_dataset_path, transform=transforms.Compose([
  #  np.array,
  #  mask2distmatrix
  #]))
  #for idx, data in enumerate(dataset):
  #  print(f'idx: {idx}')
  #  print(f'data: {data}')
  #  #torch.save(data, 'data_drive_path{}'.format(idx))
  #print(dataset)

# # Simplified version for test
# def process_png_file(mask_path, idx, output_folder='./results/reconstruction'):
#     # Perform specific action for each PNG file
#     print("Processing:", mask_path)
#     mask = plt.imread(mask_path)

#     # Get the contour
#     x, y = find_longest_contour(mask)

#     # Reinterpolate (spline)
#     x_reinterpolated, y_reinterpolated = spline_interpolation(x, y)
#     plt.scatter(x_reinterpolated, y_reinterpolated, s=6)
#     plt.savefig(f'{output_folder}/original_contour{idx}.png')
#     plt.clf()

#     # Build the distance matrix
#     dm = build_distance_matrix(x_reinterpolated, y_reinterpolated)
#     np.save(f"{output_folder}/matrix_{idx}.npy", dm)

#     # Reconstruction coordinates and matrix (MDS)
#     reconstructed_coords = dist_to_coords(dm)
#     print(reconstructed_coords)
#     plt.scatter(*zip(*reconstructed_coords), s=6)
#     plt.savefig(f'{output_folder}/reconstructed_contour{idx}.png')
#     plt.clf()
#     reconstructed_matrix = euclidean_distances(reconstructed_coords)

#     # Error with matrix
#     err = np.average(dm - reconstructed_matrix)
#     print(f"Dist error is: {err}")

###############################################################################

params = types.SimpleNamespace(**{
    "mask_dataset_path": None
  , "output_path": None
  , "raw_sampling_sparsity": 1
  , "spline_sampling": 512
})

if __name__ == "__main__":

  def auto_pos_int (x):
    val = int(x,0)
    if val <= 0:
        raise argparse.ArgumentTypeError("argument must be a positive int. Got {:d}.".format(val))
    return val

  parser = argparse.ArgumentParser(description='Turn mask dataset into distance matrix dataset')

  parser.add_argument('path', metavar='PATH', help=f"The PATH to the dataset")
  parser.add_argument('-o', '--output-path', help="The desired output path to the generated dataset")
  parser.add_argument('-s', '--raw-sampling-sparsity', type=auto_pos_int
    , help=f"The desired sparsity (in number of points) when sampling the raw contour (default, every {params.raw_sampling_sparsity} point(s))")
  parser.add_argument('-n', '--spline-sampling', type=auto_pos_int
    , help=f"The desired number of points when sampling the spline contour (default, {params.spline_sampling} point(s))")
  parser.add_argument('-v', '--verbose', action='count', default=0
    , help="Increase verbosity level by adding more \"v\".")

  # parse command line arguments
  clargs=parser.parse_args()

  # set verbosity level for vprint function
  vprint.lvl = clargs.verbose

  # update default params with clargs
  if clargs.path:
    params.mask_dataset_path = clargs.path
  #params.mask_dataset_path = "/nfs/research/uhlmann/afoix/tiny_synthcellshapes_dataset"
  if clargs.output_path:
    params.output_path = clargs.output_path
  if clargs.raw_sampling_sparsity:
    params.raw_sampling_sparsity = clargs.raw_sampling_sparsity
  if clargs.spline_sampling:
    params.spline_sampling = clargs.spline_sampling

  masks2distmatrices(params)



###############################################################################
###############################################################################
###############################################################################
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
