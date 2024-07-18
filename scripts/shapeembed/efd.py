#! /usr/bin/env python3

import os
import types
import pyefd
import logging
import argparse

# own imports
from evaluation import *

def run_elliptic_fourier_descriptors(dataset_params, contour_size, logger):
  # access the dataset
  assert dataset_params.type == 'mask'
  ds = datasets.ImageFolder( dataset_params.path
                           , transform=transforms.Compose([
                               transforms.Grayscale(1)
                             , ImageToCoords(contour_size) ]))
  # ... and run efd on each image
  dfs = []
  logger.info(f'running efd on {dataset_params.name}')
  logger.info(f'({dataset_params.path})')
  for i, (img, lbl) in enumerate(tqdm.tqdm(ds)):
    coeffs = pyefd.elliptic_fourier_descriptors(img, order=10, normalize=False)
    norm_coeffs = pyefd.normalize_efd(coeffs)
    df = pandas.DataFrame({
      "norm_coeffs": norm_coeffs.flatten().tolist()
    , "coeffs": coeffs.flatten().tolist()
    }).T.rename_axis("coeffs")
    df['class'] = lbl
    df.set_index("class", inplace=True, append=True)
    dfs.append(df)
  # concatenate results as a single dataframe and return it
  df = pandas.concat(dfs).xs('coeffs', level='coeffs')
  df.reset_index(level='class', inplace=True)
  return df

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run efd on a given dataset')
  
  dflt_dataset=('tiny_synthetic_shapes', '/nfs/research/uhlmann/afoix/datasets/image_datasets/tiny_synthetic_shapes', 'mask')
  parser.add_argument(
      '-d', '--dataset', nargs=3, metavar=('NAME', 'PATH', 'TYPE'), default=dflt_dataset
    , help=f"The NAME, PATH and TYPE of the dataset (default: {dflt_dataset})")

  dflt_contour_size=512

  parser.add_argument(
      '-o', '--output-dir', metavar='OUTPUT_DIR', default='./'
    , help=f"The OUTPUT_DIR path to use to dump results")

  parser.add_argument('-v', '--verbose', action='count', default=0
    , help="Increase verbosity level by adding more \"v\".")

  # parse command line arguments
  clargs=parser.parse_args()

  # set verbosity level
  logger = logging.getLogger(__name__)
  if clargs.verbose > 2:
    logger.setLevel(logging.DEBUG)
  elif clargs.verbose > 0:
    logger.setLevel(logging.INFO)

  # update default params with clargs
  dataset = types.SimpleNamespace( name=clargs.dataset[0]
                                 , path=clargs.dataset[1]
                                 , type=clargs.dataset[2] )
  contour_size = dflt_contour_size

  # create output dir if it does not exist
  os.makedirs(clargs.output_dir, exist_ok=True)

  # efd on input data and score

  efd_df = run_elliptic_fourier_descriptors(dataset, contour_size, logger)

  logger.info(f'-- efd on {dataset.name}, raw\n{efd_df}')
  efd_df.to_csv(f"{clargs.output_dir}/{dataset.name}_efd_df.csv")
  umap_plot(efd_df, f'{dataset.name}_efd', outputdir=clargs.output_dir)

  efd_cm, efd_score_df = score_dataframe(efd_df, 'efd')

  logger.info(f'-- efd on {dataset.name}, score\n{efd_score_df}')
  efd_score_df.to_csv(f"{clargs.output_dir}/{dataset.name}_efd_score_df.csv")
  logger.info(f'-- confusion matrix:\n{efd_cm}')
  confusion_matrix_plot(efd_cm, f'{dataset.name}_efd', clargs.output_dir)
