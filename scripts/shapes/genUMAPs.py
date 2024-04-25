#! /usr/bin/env python3

import os
import os.path
import pandas as pd
import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
import bokeh.plotting
import argparse
import datetime
import pathlib
import multiprocessing
import subprocess

# Seed everything
np.random.seed(42)

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

# render UMAPS
def render_umap_core(df, output_dir, n_neighbors, min_dist, n_components):
  name = f'umap_{n_neighbors}_{min_dist}_{n_components}'
  vprint(4, f'generate {name}')
  vprint(5, f'n_neigbhors: {type(n_neighbors)} {n_neighbors}')
  vprint(5, f'min_dist: {type(min_dist)} {min_dist}')
  vprint(5, f'n_components: {type(n_components)} {n_components}')
  umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
  mapper = umap_model.fit(df.drop(['class_idx','class','fname'], axis=1))
  umap.plot.points(mapper, labels=np.array(df['class']))
  plt.savefig(f'{output_dir}/{name}.png')
  #p = umap.plot.interactive(mapper, labels=df['class_idx'], hover_data=df[['class','fname']])
  p = umap.plot.interactive(mapper, values=df.drop(['class_idx','class','fname'], axis=1).mean(axis=1), theme='viridis', hover_data=df[['class','fname']])
  # save interactive plot as html
  bokeh.plotting.output_file(f"{output_dir}/{name}.html")
  bokeh.plotting.save(p)

def render_umap(latent_space_pkl, output_dir, n_neighbors, min_dist, n_components):
  # create output directory if it does not already exist
  os.makedirs(output_dir, exist_ok=True)
  # load latent space
  df = pd.read_pickle(latent_space_pkl)
  # render umap
  render_umap_core(df, output_dir, n_neighbors, min_dist, n_components)

###############################################################################

if __name__ == "__main__":

  def auto_pos_int (x):
    val = int(x,0)
    if val <= 0:
        raise argparse.ArgumentTypeError("argument must be a positive int. Got {:d}.".format(val))
    return val

  parser = argparse.ArgumentParser(description='generate umaps')
    
  parser.add_argument('latent_space', metavar='LATENT_SPACE', type=os.path.abspath
                     , help=f"The path to the latent space")
  parser.add_argument('-j', '--n_jobs', type=auto_pos_int, default=2*os.cpu_count()
                     , help="number of jobs to start. Default is 2x the number of CPUs.")
  parser.add_argument('--slurm', action=argparse.BooleanOptionalAction)
  parser.add_argument('-n', '--n_neighbors', nargs='+', type=auto_pos_int, default=[50]
                     , help="A list of the number of neighbors to use in UMAP. Default is [50].")
  parser.add_argument('-m', '--min_dist', nargs='+', type=float, default=[0.8]
                     , help="A list of the minimum distances to use in UMAP. Default is [0.8].")
  parser.add_argument('-c', '--n_components', nargs='+', type=auto_pos_int, default=[2]
                     , help="A list of the number of components to use in UMAP. Default is [2].")
  parser.add_argument( '-o', '--output-dir', metavar='OUTPUT_DIR', default=f'{os.getcwd()}/umaps'
                     , help=f"The OUTPUT_DIR path to use to dump results")
  parser.add_argument('-v', '--verbose', action='count', default=0
    , help="Increase verbosity level by adding more \"v\".")

  # parse command line arguments
  clargs=parser.parse_args()

  # set verbosity level for vprint function
  vprint.lvl = clargs.verbose

  #for x,y,z in [(x, y, z) for x in clargs.n_neighbors
  #                        for y in clargs.min_dist
  #                        for z in clargs.n_components]:
  #  render_umap(df, x, y, z)

  params=[(x, y, z) for x in clargs.n_neighbors
                    for y in clargs.min_dist
                    for z in clargs.n_components]
  if clargs.slurm:
    vprint(2, f'running with slurm')
    for (n_neighbors, min_dist, n_components) in params:
      vprint(3, f'running with n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}')
      print('Directory Name: ', os.path.dirname(__file__))

      cmd = [ "srun"
            , "-t", "50:00:00"
            , "--mem=200G"
            , "--gpus=a100:1"
            ,  "--job-name", f"render_umap_{n_neighbors}_{min_dist}_{n_components}"
            , "--pty"
            , "python3", "-c"
            , f"""
import sys
sys.path.insert(1, '{os.path.dirname(__file__)}')
import genUMAPs
genUMAPs.render_umap('{clargs.latent_space}','{clargs.output_dir}',{n_neighbors},{min_dist},{n_components})
"""]
      vprint(4, cmd)
      subprocess.run(cmd)

  else:
    vprint(2, f'running with python multiprocessing')

    # create output directory if it does not already exist
    os.makedirs(clargs.output_dir, exist_ok=True)

    # load latent space
    df = pd.read_pickle(clargs.latent_space)

    def render_umap_wrapper(args):
      render_umap(df, clargs.output_dir, *args)
    with multiprocessing.Pool(clargs.n_jobs) as pool:
      pool.starmap(render_umap_wrapper, params)