#! /usr/bin/env python3

import os
import types
import pyefd
import logging
import argparse

# own imports
#import bioimage_embed # necessary for the datamodule class to make sure we get the same test set
from bioimage_embed.shapes.transforms import ImageToCoords
from evaluation import *

def get_dataset(dataset_params):
  # access the dataset
  assert dataset_params.type == 'mask', f'unsupported dataset type {dataset_params.type}'
  dataset = datasets.ImageFolder( dataset_params.path
                                , transform=transforms.Compose([
                                    transforms.Grayscale(1)
                                  , ImageToCoords(contour_size) ]))
  return dataset
  #dataloader = bioimage_embed.lightning.DataModule(dataset, shuffle=True)
  #dataloader.setup()
  #return dataloader.test

def run_elliptic_fourier_descriptors(dataset, contour_size, logger):
  # run efd on each image
  dfs = []
  logger.info(f'running efd on {dataset}')
  for i, (img, lbl) in enumerate(tqdm.tqdm(dataset)):
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

  efd_df = run_elliptic_fourier_descriptors(get_dataset(dataset), contour_size, logger)

  logger.info(f'-- efd on {dataset.name}, raw\n{efd_df}')
  efd_df.to_csv(f"{clargs.output_dir}/{dataset.name}-efd-raw_df.csv")
  umap_plot(efd_df, f'{dataset.name}-efd', outputdir=clargs.output_dir)

  efd_cm, efd_score_df = score_dataframe(efd_df, 'efd')

  logger.info(f'-- efd on {dataset.name}, score\n{efd_score_df}')
  efd_score_df.to_csv(f"{clargs.output_dir}/{dataset.name}-efd-score_df.csv")
  logger.info(f'-- confusion matrix:\n{efd_cm}')
  confusion_matrix_plot(efd_cm, f'{dataset.name}-efd', clargs.output_dir)
