#! /usr/bin/env python3

import os
import logging
import argparse
import datetime
import subprocess

# shapeembed parameters to sweap
################################################################################

datasets_pfx = '/nfs/research/uhlmann/afoix/datasets/image_datasets'
datasets = [
#  ("tiny_synthcell", f"{datasets_pfx}/tiny_synthcellshapes_dataset/", "mask")
  ("vampire", f"{datasets_pfx}/vampire/torchvision/Control/", "mask")
#, ("bbbc010", f"{datasets_pfx}/bbbc010/BBBC010_v1_foreground_eachworm/", "mask")
, ("synthcell", f"{datasets_pfx}/synthcellshapes_dataset/", "mask")
#, ("helakyoto", f"{datasets_pfx}/H2b_10x_MD_exp665/samples/", "mask")
#, ("allen", f"{datasets_pfx}/allen_dataset/", "mask")
]

models = [
  "resnet18_vae"
#, "resnet50_vae"
#, "resnet18_beta_vae"
#, "resnet18_vae_bolt"
#, "resnet50_vae_bolt"
, "resnet18_vqvae"
#, "resnet50_vqvae"
#, "resnet18_vqvae_legacy"
#, "resnet50_vqvae_legacy"
#, "resnet101_vqvae_legacy"
#, "resnet110_vqvae_legacy"
#, "resnet152_vqvae_legacy"
#, "resnet18_vae_legacy"
#, "resnet50_vae_legacy"
]

model_params = {
  "resnet18_beta_vae": {'beta': [0.5, 1.0, 2]}
}

compression_factors = [2, 4]

batch_sizes = [4]

# other parameters
################################################################################

dflt_slurm_dir=f'{os.getcwd()}/slurm_info_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
dflt_out_dir=f'{os.getcwd()}/output_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

slurm_time = '50:00:00'
slurm_mem = '200G'
slurm_gpus = 'a100:1'

n_epochs = 2

wandb_project='shapeembed'

slurm_script="""#! /bin/bash
echo "running shape embed with:"
echo "  - dataset {dataset[0]} ({dataset[1]}, {dataset[2]})"
echo "  - model {model} ({model_params})"
echo "  - compression_factor {compression_factor}"
echo "  - batch size {batch_size}"
python3 shapeembed.py --wandb-project {wandb_project} --num-epochs {n_epochs} --dataset {dataset[0]} {dataset[1]} {dataset[2]} --model {model} --compression-factor {compression_factor} --batch-size {batch_size} --clear-checkpoints --output-dir {out_dir}
"""

################################################################################

def spawn_slurm_job(logger, slurm_out_dir, out_dir, dataset, model, compression_factor, batch_size):
  jobname = f'shapeembed_{dataset[0]}_{model}_{compression_factor}_{batch_size}'
  logger.info(f'spawning {jobname}')
  with open(f'{slurm_out_dir}/{jobname}.script', mode='w+') as fp:
    fp.write(slurm_script.format( dataset=dataset
                                , model=model
                                , model_params=[]
                                , compression_factor=compression_factor
                                , batch_size=batch_size
                                , out_dir=out_dir
                                , wandb_project=wandb_project
                                , n_epochs=n_epochs ))
    fp.flush()
    logger.info(f'written {fp.name}')
    logger.debug(f'cat {fp.name}')
    result = subprocess.run(['cat', fp.name], stdout=subprocess.PIPE)
    logger.debug(result.stdout.decode('utf-8'))
    result = subprocess.run([ 'sbatch'
                            , '--time', slurm_time
                            , '--mem', slurm_mem
                            , '--job-name', jobname
                            , '--output', f'{slurm_out_dir}/{jobname}.out'
                            , '--error', f'{slurm_out_dir}/{jobname}.err'
                            #, '--gres', n_gpus(ls)
                            , f'--gpus={slurm_gpus}'
                            , fp.name ], stdout=subprocess.PIPE)
    logger.info(result.stdout.decode('utf-8'))

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

  for params in [ (ds, m, cf, bs) for ds in datasets
                                  for m in models
                                  for cf in compression_factors
                                  for bs in batch_sizes ]:
    spawn_slurm_job(logger, clargs.slurm_output_dir, clargs.output_dir, *params)
