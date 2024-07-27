#! /usr/bin/env python3

import os
import types
import logging
import argparse
from skimage import measure

# own imports
#import bioimage_embed # necessary for the datamodule class to make sure we get the same test set
from evaluation import *

def get_dataset(dataset_params):
  # access the dataset
  assert dataset_params.type == 'mask', f'unsupported dataset type {dataset_params.type}'
  dataset = datasets.ImageFolder(dataset_params.path, transforms.Grayscale(1))
  return dataset
  #dataloader = bioimage_embed.lightning.DataModule(dataset, shuffle=True)
  #dataloader.setup()
  #return dataloader.test

def run_regionprops( dataset
                   , properties
                   , logger ):
  # run regionprops for the given properties for each image
  dfs = []
  logger.info(f'running regionprops on {dataset}')
  for i, (img, lbl) in enumerate(tqdm.tqdm(dataset)):
    data = numpy.where(numpy.array(img)>20, 255, 0)
    t = measure.regionprops_table(data, properties=properties)
    df = pandas.DataFrame(t)
    assert df.shape[0] == 1, f'More than one object in image #{i}'
    df.index = [i]
    df['class'] = lbl
    #df.set_index("class", inplace=True)
    dfs.append(df)
  # concatenate results as a single dataframe and return it
  df = pandas.concat(dfs)
  return df

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run regionprops on a given dataset')
  
  dflt_dataset=('tiny_synthetic_shapes', '/nfs/research/uhlmann/afoix/datasets/image_datasets/tiny_synthetic_shapes', 'mask')
  parser.add_argument(
      '-d', '--dataset', nargs=3, metavar=('NAME', 'PATH', 'TYPE'), default=dflt_dataset
    , help=f"The NAME, PATH and TYPE of the dataset (default: {dflt_dataset})")

  dflt_properties=[ "area"
                  , "perimeter"
                  , "centroid"
                  , "major_axis_length"
                  , "minor_axis_length"
                  , "orientation" ]

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
  properties = dflt_properties

  # create output dir if it does not exist
  os.makedirs(clargs.output_dir, exist_ok=True)

  # regionprops on input data and score

  regionprops_df = run_regionprops(get_dataset(dataset), properties, logger)

  logger.info(f'-- regionprops on {dataset.name}, raw\n{regionprops_df}')
  regionprops_df.to_csv(f"{clargs.output_dir}/{dataset.name}-regionprops-raw_df.csv")
  umap_plot(regionprops_df, f'{dataset.name}-regionprops', outputdir=clargs.output_dir)

  regionprops_cm, regionprops_score_df = score_dataframe(regionprops_df, 'regionprops')

  logger.info(f'-- regionprops on {dataset.name}, score\n{regionprops_score_df}')
  regionprops_score_df.to_csv(f"{clargs.output_dir}/{dataset.name}-regionprops-score_df.csv")
  logger.info(f'-- confusion matrix:\n{regionprops_cm}')
  confusion_matrix_plot(regionprops_cm, f'{dataset.name}-regionprops', clargs.output_dir)
