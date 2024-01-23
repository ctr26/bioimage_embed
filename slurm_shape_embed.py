#! /usr/bin/env python3

import os
import subprocess
import tempfile

## Assign the arguments to variables
#model_arg=$1
#sizes_list="${@:2}"
#
## Create SLURM job script
#job_script="slurm_job.sh"
#
#echo "#!/bin/bash" > "$job_script"
#echo "#SBATCH --job-name=ite_shape_embed" >> "$job_script"
#echo "#SBATCH --output=ite_shape_embed.out" >> "$job_script"
#echo "#SBATCH --error=ite_shape_embed.err" >> "$job_script"
#echo "#SBATCH --gres=gpu:2" >> "$job_script"  # Adjust the number of CPUs as needed
#echo "#SBATCH --mem=50GB" >> "$job_script"          # Adjust the memory requirement as needed
#echo "" >> "$job_script"
#
## Loop through the sizes and append the Python command to the job script
#for size in $sizes_list; do
#    echo "python ite_shape_embed.py --model $model_arg --ls_size $size" >> "$job_script"
#done
#
## Submit SLURM job
#sbatch "$job_script"

models = [
  "resnet18_vae"
, "resnet50_vae"
, "resnet18_vae_bolt"
, "resnet50_vae_bolt"
, "resnet18_vqvae"
, "resnet50_vqvae"
, "resnet18_vqvae_legacy"
, "resnet50_vqvae_legacy"
, "resnet101_vqvae_legacy"
, "resnet110_vqvae_legacy"
, "resnet152_vqvae_legacy"
, "resnet18_vae_legacy"
, "resnet50_vae_legacy"
]
batch_sizes = [4, 8, 16]
latent_space_sizes = [64, 128, 256, 512]

slurm_script="""#!/bin/bash

JOB_NAME=shape_embed_{model}_{b_size}_{ls_size}
echo "running shape embed with:"
echo "  - model {model}"
echo "  - batch size {b_size}"
echo "  - latent space size {ls_size}"
rand_name=$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 16)
mkdir -p slurm_rundir/$rand_name
cp -r $(ls | grep -v slurm_rundir) slurm_rundir/$rand_name/.
cd slurm_rundir/$rand_name
python3 scripts/shapes/shape_embed.py --model {model} --batch-size {b_size} --latent-space-size {ls_size} --clear-checkpoints
"""

def mem_size(ls):
    if ls <= 128:
        return '50GB'
    if ls > 128:
        return '100GB'
    if ls > 256:
        return '300GB'

def n_gpus(ls):
    if ls <= 128:
        return 'gpu:2'
    if ls > 128:
        return 'gpu:2'
    if ls > 256:
        return 'gpu:3'

if __name__ == "__main__":
    
    slurmdir = f'{os.getcwd()}/slurmdir'
    os.makedirs(slurmdir, exist_ok=True)
    for m, bs, ls in [ (m,bs,ls) for  m in models
                                 for bs in batch_sizes
                                 for ls in latent_space_sizes ]:
        jobname = f'shape_embed_{m}_{bs}_{ls}'
        print(jobname)
        fp = open(mode='w+', file=f'{slurmdir}/slurm_script_shape_embed_{m}_{bs}_{ls}.script')
        fp.write(slurm_script.format(model=m, b_size=bs, ls_size=ls))
        fp.flush()
        print(f'{fp.name}')
        print(f'cat {fp.name}')
        result = subprocess.run(['cat', fp.name], stdout=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
        print(mem_size(ls))
        result = subprocess.run([ 'sbatch'
                                , '--time', '10:00:00'
                                , '--mem', mem_size(ls)
                                , '--job-name', jobname
                                , '--output', f'{slurmdir}/{jobname}.out'
                                , '--error', f'{slurmdir}/{jobname}.err'
                                , '--gres', n_gpus(ls)
                                , fp.name], stdout=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
