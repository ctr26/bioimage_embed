
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np
import argparse
import pathlib
import types
import glob

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

def asym_to_sym(asym_dist_mat):
  return np.max(np.stack([asym_dist_mat, asym_dist_mat.T]), axis=0)

def dist_to_coords(dst_mat):
  embedding = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto')
  return embedding.fit_transform(dst_mat)

def distmatrices2contour(params):
  plt.clf()
  dm_npys = glob.glob(f'{params.matrices_folder}/*.npy')
  for dm_npy in dm_npys:
    dm = np.load(dm_npy)
    vprint(2, f'{dm_npy}: dm.shape={dm.shape}')
    dm = asym_to_sym(dm)
    p = pathlib.Path(dm_npy)
    p = p.with_suffix('.png')
    reconstructed_coords = dist_to_coords(dm)
    plt.axes().set_aspect('equal')
    plt.scatter(*zip(*reconstructed_coords), s=6)
    plt.savefig(p)
    vprint(2, f'saved {p}')
    plt.clf()

###############################################################################

params = types.SimpleNamespace(**{
    "matrices_folder": None
})

if __name__ == "__main__":

  def auto_pos_int (x):
    val = int(x,0)
    if val <= 0:
        raise argparse.ArgumentTypeError("argument must be a positive int. Got {:d}.".format(val))
    return val

  parser = argparse.ArgumentParser(description='Turn distance matrices into contours')

  parser.add_argument('matrices_folder', metavar='MATRICES_FOLDER', help=f"The path to the matrices folder")
  parser.add_argument('-v', '--verbose', action='count', default=0
    , help="Increase verbosity level by adding more \"v\".")

  # parse command line arguments
  clargs=parser.parse_args()

  # set verbosity level for vprint function
  vprint.lvl = clargs.verbose

  # update default params with clargs
  params.matrices_folder = clargs.matrices_folder

  distmatrices2contour(params)
