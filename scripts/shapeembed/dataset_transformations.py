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
import logging

# logging facilities
###############################################################################
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# misc helpers
###############################################################################

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

# API functions
###############################################################################

def find_longest_contour(mask, normalise_coord=False):
  """Find all contours existing in 'mask' and return the longest one

     :param mask: The image with masked objects
     :param normalise_coord(default: False): optionally normalise coordinates

     :returns: the longest contour as a pair of lists for the x and y
               coordinates
  """
  # force the image to grayscale
  if len(mask.shape) == 3: # (lines, columns, number of channels)
    mask = rgb2grey(mask)
  # extract the contours from the now grayscale image
  contours = sk.measure.find_contours(mask, 0.8)
  logger.debug(f'find_longest_contour: len(contours) {len(contours)}')
  # sort the contours by length
  contours = sorted(contours, key=lambda x: len(x), reverse=True)
  # isolate the longest contour (first in the sorted list)
  x, y = contours[0][:, 0], contours[0][:, 1]
  # optionally normalise the coordinates in the countour
  if normalise_coord:
    x = x - np.min(x)
    x = x / np.max(x)
    y = y - np.min(y)
    y = y / np.max(y)
  # return the contour as a pair of lists of x and y coordinates
  return x, y

def spline_interpolation(x, y, spline_sampling, raw_sampling_sparsity=1):
  """Return a resampled spline interpolation of a provided contour

     :param x: The list of x coordinates of a contour
     :param y: The list of y coordinates of a contour
     :param spline_sampling: The number of points to sample on the spline
     :param raw_sampling_sparsity (default=1):
       The distance (in number of gaps) to the next point to consider in the
       raw contour (i.e. whether consider every point, every other point
       , every 3 points... This might be considered to avoid artifacts due to
       high point count contours over low pixel resolution images, with contour
       effectively curving around individual pixel edges)

     :returns: the resampled spline with spline_sampling points as a pair of
               lists of x and y coordinates
  """
  # Force sparsity to be at least one
  raw_sampling_sparsity = max(1, raw_sampling_sparsity)
  logger.debug(f'spline_interpolation: running with raw_sampling_sparsity {raw_sampling_sparsity} and spline_sampling {spline_sampling}')
  logger.debug(f'spline_interpolation: x.shape {x.shape} y.shape {y.shape}')
  # prepare the spline interpolation of the given contour
  tck, u = splprep( [x[::raw_sampling_sparsity], y[::raw_sampling_sparsity]]
                  , s = 0 # XXX
                  , per = True # closed contour (periodic spline)
                  )
  # how many times to sample the spline
  # last parameter is how dense is our spline, how many points.
  new_u = np.linspace(u.min(), u.max(), spline_sampling)
  # evaluate and return the sampled spline
  x_spline, y_spline = splev(new_u, tck)
  return x_spline, y_spline

def build_distance_matrix(x_reinterpolated, y_reinterpolated):
  """Turn a (reinterpolated) contour into a distance matrix

     :param x_reinterpolated: The list of x coordinates of a contour
     :param y_reinterpolated: The list of y coordinates of a contour

     :returns: the distance matrix characteristic of the provided contour
  """
  # reshape the pair of lists of individual x and y coordinates as a single
  # numpy array of pairs of (x,y) coordinates
  reinterpolated_contour = np.column_stack([ x_reinterpolated
                                           , y_reinterpolated ])
  # build the distance matrix from the reshaped input data
  dm = scipy.spatial.distance_matrix( reinterpolated_contour
                                    , reinterpolated_contour )
  return dm

def dist_to_coords(dst_mat):
  """Turn a distance matrix into the corresponding contour
     XXX
     TODO sort out exactly the specifics here...
  """
  embedding = MDS(n_components=2, dissimilarity='precomputed')
  return embedding.fit_transform(dst_mat)

def mask2distmatrix(mask, matrix_size=512, raw_sampling_sparsity=1):
  """Get the distance matrix characteristic of the (biggest) object in the
     provided image

     :param mask: The image with masked objects
     :param matrix_size(default: 512): the desired matrix size
     :param raw_sampling_sparsity (default=1):
       The distance (in number of gaps) to the next point to consider in the
       raw contour (i.e. whether consider every point, every other point
       , every 3 points... This might be considered to avoid artifacts due to
       high point count contours over low pixel resolution images, with contour
       effectively curving around individual pixel edges)

     :returns: the distance matrix characteristic of the (biggest) object in
               the provided image
  """
  logger.debug(f'mask2distmatrix: running with raw_sampling_sparsity {raw_sampling_sparsity} and matrix_size {matrix_size}')
  # extract mask contour
  x, y = find_longest_contour(mask, normalise_coord=True)
  logger.debug(f'mask2distmatrix: found contour shape x {x.shape} y {y.shape}')
  # Reinterpolate (spline)
  x_reinterpolated, y_reinterpolated = spline_interpolation(x, y, matrix_size, raw_sampling_sparsity)
  # Build the distance matrix
  dm = build_distance_matrix(x_reinterpolated, y_reinterpolated)
  logger.debug(f'mask2distmatrix: created distance matrix shape {dm.shape}')
  return dm
