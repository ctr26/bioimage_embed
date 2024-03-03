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
, "resnet18_vqvae"
, "resnet18_vqvae_legacy"
, "resnet18_vae_legacy"
]
batch_sizes = [4]
latent_space_sizes = [128]

datasets = [
#  ("tiny_synthcell", "tiny_synthcellshapes_dataset/")
  ("vampire", "vampire/torchvision/Control/")
, ("bbbc010", "bbbc010/BBBC010_v1_foreground_eachworm/")
, ("synthcell", "synthcellshapes_dataset/")
, ("helakyoto", "H2b_10x_MD_exp665/samples/")
, ("allen", "allen_dataset/")
]

wandb_project='shape-embed-test-changes'

slurm_script="""#!/bin/bash

echo "running shape embed with:"
echo "  - model {model}"
echo "  - dataset {dataset[0]} ({dataset[1]})"
echo "  - batch size {b_size}"
echo "  - latent space size {ls_size}"
rand_name=$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 16)
mkdir -p slurm_rundir/$rand_name
cp -r $(ls | grep -v slurm_rundir) slurm_rundir/$rand_name/.
cd slurm_rundir/$rand_name
python3 scripts/shapes/shape_embed.py --wandb-project {wandb_project} --model {model} --dataset {dataset[0]} {dataset[1]} --batch-size {b_size} --latent-space-size {ls_size} --clear-checkpoints
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
    for m, bs, ls, ds in [ (m,bs,ls,ds) for  m in models
                                 for bs in batch_sizes
                                 for ls in latent_space_sizes
                                 for ds in datasets ]:
        jobname = f'shape_embed_{m}_{ds[0]}_{bs}_{ls}'
        print(jobname)
        fp = open(mode='w+', file=f'{slurmdir}/slurm_script_shape_embed_{m}_{bs}_{ls}.script')
        fp.write(slurm_script.format(model=m, dataset=ds, b_size=bs, ls_size=ls, wandb_project=wandb_project))
        fp.flush()
        print(f'{fp.name}')
        print(f'cat {fp.name}')
        result = subprocess.run(['cat', fp.name], stdout=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
        print(mem_size(ls))
        result = subprocess.run([ 'sbatch'
                                , '--time', '24:00:00'
                                , '--mem', mem_size(ls)
                                , '--job-name', jobname
                                , '--output', f'{slurmdir}/{jobname}.out'
                                , '--error', f'{slurmdir}/{jobname}.err'
                                , '--gres', n_gpus(ls)
                                , fp.name], stdout=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
