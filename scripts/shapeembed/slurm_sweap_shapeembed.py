#! /usr/bin/env python3

import os
import glob
import copy
import types
import logging
import tempfile
import argparse
import datetime
import itertools
import subprocess

import shapeembed

# shapeembed parameters to sweap
################################################################################

datasets_pfx = '/nfs/research/uhlmann/afoix/datasets/image_datasets'
datasets = [
  ("synthetic_shapes", f"{datasets_pfx}/synthetic_shapes/", "mask")
#  ("tiny_synthcell", f"{datasets_pfx}/tiny_synthcellshapes_dataset/", "mask")
#  ("vampire", f"{datasets_pfx}/vampire/torchvision/Control/", "mask")
#, ("bbbc010", f"{datasets_pfx}/bbbc010/BBBC010_v1_foreground_eachworm/", "mask")
#, ("synthcell", f"{datasets_pfx}/synthcellshapes_dataset/", "mask")
#, ("helakyoto", f"{datasets_pfx}/H2b_10x_MD_exp665/samples/", "mask")
#, ("allen", f"{datasets_pfx}/allen_dataset/", "mask")
]

models = [
  "resnet18_vae"
, "resnet50_vae"
, "resnet18_beta_vae"
, "resnet50_beta_vae"
#, "resnet18_vae_bolt"
#, "resnet50_vae_bolt"
, "resnet18_vqvae"
, "resnet50_vqvae"
#, "resnet18_vqvae_legacy"
#, "resnet50_vqvae_legacy"
#, "resnet101_vqvae_legacy"
#, "resnet110_vqvae_legacy"
#, "resnet152_vqvae_legacy"
#, "resnet18_vae_legacy"
#, "resnet50_vae_legacy"
]

model_params = {
  "resnet18_beta_vae": {'beta': [1,2,5,10,20]}
, "resnet50_beta_vae": {'beta': [1,2,5,10,20]}
}

compression_factors = [1,2,3,5,10]

batch_sizes = [4, 8, 16]

def gen_params_sweap_list():
  p_sweap_list = []
  for params in [ { 'dataset': types.SimpleNamespace(name=ds[0], path=ds[1], type=ds[2])
                  , 'model_name': m
                  , 'compression_factor': cf
                  , 'latent_dim': shapeembed.compressed_n_features(512, cf)
                  , 'batch_size': bs
                  } for ds in datasets
                    for m in models
                    for cf in compression_factors
                    for bs in batch_sizes ]:
    # per model params:
    if params['model_name'] in model_params:
      mps = model_params[params['model_name']]
      for ps in [dict(zip(mps.keys(), vs)) for vs in itertools.product(*mps.values())]:
        newparams = copy.deepcopy(params)
        newparams['model_args'] = types.SimpleNamespace(**ps)
        p_sweap_list.append(types.SimpleNamespace(**newparams))
    else:
      p_sweap_list.append(types.SimpleNamespace(**params))
  return p_sweap_list

def params_match(x, ys):
  found = False
  def check_model_args(a, b):
    a_yes = hasattr(a, 'model_args')
    b_yes = hasattr(b, 'model_args')
    if not a_yes and not b_yes: return True
    if a_yes and b_yes: return a.model_args == b.model_args
    return False
  for y in ys:
    if x.dataset.name == y.dataset \
      and x.model_name == y.model_name \
      and check_model_args(x, y) \
      and x.compression_factor == y.compression_factor \
      and x.latent_dim == y.latent_dim \
      and x.batch_size == y.batch_size:
      found = True
      break
  return found

# other parameters
################################################################################

dflt_slurm_dir=f'{os.getcwd()}/slurm_info_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
dflt_out_dir=f'{os.getcwd()}/output_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

slurm_time = '50:00:00'
slurm_mem = '250G'
slurm_gpus = 'a100:1'

shapeembed_script=f'{os.getcwd()}/shapeembed.py'
wandb_project='shapeembed'

slurm_script="""#! /bin/bash
echo "running shape embed with:"
echo "  - dataset {dataset[0]} ({dataset[1]}, {dataset[2]})"
echo "  - model {model} ({model_params})"
echo "  - compression_factor {compression_factor}"
echo "  - batch size {batch_size}"
python3 shapeembed.py --wandb-project {wandb_project} --dataset {dataset[0]} {dataset[1]} {dataset[2]} --model {model} --compression-factor {compression_factor} --batch-size {batch_size} --clear-checkpoints --output-dir {out_dir} {extra_args}
"""

################################################################################

def model_params_from_model_params_str(modelparamsstr):
  rawps = modelparamsstr.split('_')
  ps = {}
  for p in rawps:
    if p[0:4] == 'beta': ps['beta'] = float(p[4:])
  return types.SimpleNamespace(**ps)

def params_from_job_str(jobstr):
  raw = jobstr.split('-')
  ps = {}
  ps['batch_size'] = int(raw.pop())
  ps['latent_dim'] = int(raw.pop())
  ps['compression_factor'] = int(raw.pop())
  if len(raw) == 3:
    ps['model_args'] = model_params_from_model_params_str(raw.pop())
  ps['model_name'] = raw.pop()
  ps['dataset'] = raw.pop()
  return types.SimpleNamespace(**ps)

def find_done_params(out_dir):
  ps = []
  for f in glob.glob(f'{out_dir}/*-shapeembed-score_df.csv'):
    ps.append(params_from_job_str(os.path.basename(f)[:-24]))
  return ps

def spawn_slurm_job(slurm_out_dir, out_dir, ps, logger=logging.getLogger(__name__)):

  jobname = shapeembed.job_str(ps)
  cmd = [ 'python3', shapeembed_script
        , '--wandb-project', wandb_project
        , '--output-dir', out_dir
        ]
  cmd += [ '--clear-checkpoints'
         , '--no-early-stop'
         , '--num-epochs', 150
         ]
  cmd += [ '--dataset', ps.dataset.name, ps.dataset.path, ps.dataset.type
         , '--model', ps.model_name
         , '--compression-factor', ps.compression_factor
         , '--batch-size', ps.batch_size
         ]
  if hasattr(ps, 'model_args'):
    for k, v in vars(ps.model_args).items():
      cmd.append(f'--model-arg-{k}')
      cmd.append(f'{v}')
  logger.debug(" ".join(map(str,cmd)))
  with tempfile.NamedTemporaryFile('w+') as fp:
    fp.write('#! /usr/bin/env sh\n')
    fp.write(" ".join(map(str,cmd)))
    fp.write('\n')
    fp.flush()
    cmd = [ 'sbatch'
          , '--time', slurm_time
          , '--mem', slurm_mem
          , '--job-name', jobname
          , '--output', f'{slurm_out_dir}/{jobname}.out'
          , '--error', f'{slurm_out_dir}/{jobname}.err'
          , f'--gpus={slurm_gpus}'
          , fp.name ]
    logger.debug(" ".join(map(str,cmd)))
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    logger.debug(result.stdout.decode('utf-8'))
  logger.info(f'job spawned for {ps}')


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Sweap parameters for shapeembed')
  
  parser.add_argument(
      '-s', '--slurm-output-dir', metavar='SLURM_OUTPUT_DIR', default=dflt_slurm_dir
    , help=f"The SLURM_OUTPUT_DIR path to use to dump slurm info")
  
  parser.add_argument(
      '-o', '--output-dir', metavar='OUTPUT_DIR', default=dflt_out_dir
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

  os.makedirs(clargs.slurm_output_dir, exist_ok=True)
  os.makedirs(clargs.output_dir, exist_ok=True)

  done_params = find_done_params(clargs.output_dir)
  all_params  = gen_params_sweap_list()
  todo_params = [x for x in all_params if not params_match(x, done_params)]
  for ps in todo_params:
    spawn_slurm_job(clargs.slurm_output_dir, clargs.output_dir, ps, logger=logger)
