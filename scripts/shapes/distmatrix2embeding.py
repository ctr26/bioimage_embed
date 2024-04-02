from torchvision import datasets, transforms
import pytorch_lightning as pl
import numpy as np
import bioimage_embed
import bioimage_embed.shapes
import bioimage_embed.lightning
from bioimage_embed.lightning import DataModule
import argparse
import datetime
import torch
import types

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

def maybe_roll (dist_mat, p = 0.5):
  if np.random.rand() < p:
    return np.roll(dist_mat, np.random.randint(0, dist_mat.shape[0]), (0,1))
  else:
    return dist_mat

def sanity_check (dist_mat):
  if not np.allclose(dist_mat, dist_mat.T):
    raise ValueError("Matrix is not symmetric")
  if np.any(dist_mat < 0):
    raise ValueError("Matrix has negative values")
  if np.any(np.diag(dist_mat)):
    raise ValueError("Matrix has non-zero diagonal")
  return dist_mat

# Main process
###############################################################################

def main_process(params):

    # Loading the data (matrices)
    ###########################################################################

    preproc_transform = transforms.Compose([
        lambda x: x / np.linalg.norm(x, "fro"), # normalize the matrix
        lambda x: maybe_roll(x, p = 1.0), # "potentially" roll the matrix
        sanity_check, # check if the matrix is symmetric and positive, and the diagonal is zero
        torch.as_tensor, # turn (H,W) numpy array into a (H,W) tensor
        lambda x: x.repeat(3, 1, 1) # turn (H,W) tensor into a (3,H,W) tensor (to fit downstream model expectations)
    ])
    dataset = datasets.DatasetFolder(params.dataset[1], loader=np.load, extensions=('npy'), transform = preproc_transform)
    #dataset = datasets.DatasetFolder(params.dataset[1], loader=lambda x: np.load(x, allow_pickle=True), extensions=('npy'), transform = preproc_transform)
    dataloader = bioimage_embed.lightning.DataModule(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )
    dataloader.setup()
    vprint(1, f'dataloader ready')

    # Build the model
    ###########################################################################

    model = bioimage_embed.models.create_model(
        model=params.model,
        input_dim=params.input_dim,
        latent_dim=params.latent_dim,
        pretrained=params.pretrained,
    )
    lit_model = bioimage_embed.shapes.MaskEmbed(model, params)
    vprint(1, f'model ready')

    # Train the model
    ###########################################################################
    
    trainer = pl.Trainer(
        #TODO logger=[wandblogger, tb_logger],
        gradient_clip_val=0.5,
        enable_checkpointing=False,
        devices=1,
        accelerator="gpu",
        accumulate_grad_batches=4,
        #TODO callbacks=[checkpoint_callback],
        min_epochs=50,
        max_epochs=params.epochs,
        log_every_n_steps=1,
    )
    trainer.fit(lit_model, datamodule=dataloader)
    lit_model.eval()
    vprint(1, f'trainer fitted')
    
    #TODO: Validate the model
    ########################################################################### 
    vprint(1, f'TODO: Validate the model')
    validation = trainer.validate(lit_model, datamodule=dataloader)
    
    #TODO: Test the model
    ###########################################################################  
    vprint(1, f'TODO: Test the model')
    testing = trainer.test(lit_model, datamodule=dataloader)
    
    # Inference on full dataset
    dataloader = DataModule(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=params.num_workers,
        # Transform is commented here to avoid augmentations in real data
        # HOWEVER, applying the transform multiple times and averaging the results might produce better latent embeddings
        # transform=transform,
    )
    dataloader.setup()
    
    predictions = trainer.predict(lit_model, datamodule=dataloader)
    
    #TODO: Pull the embedings
    ###########################################################################
    vprint(1, f'TODO: pull the embedings')
    # Use the namespace variables
    latent_space = torch.stack([d.out.z.flatten() for d in predictions])
    # Save the latent space
    np.save(f'{params.dataset[0]}_{str(datetime.datetime.now()).replace(" ", "_")}.npy', latent_space)

# default parameters
###############################################################################

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

params = types.SimpleNamespace(**{
    # general params
    "model": "resnet18_vae",
    "epochs": 150,
    "batch_size": 4,
    "num_workers": 2**4,
    "input_dim": (3, 512, 512),
    "latent_dim": 512,
    "num_embeddings": 512,
    "num_hiddens": 512,
    "pretrained": True,
    "commitment_cost": 0.25,
    "decay": 0.99,
    "frobenius_norm": False,
    "dataset": ("tiny_dist", "/nfs/research/uhlmann/afoix/tiny_synthcellshapes_dataset_distmat"),
    # optimizer_params
    "opt": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "momentum": 0.9,
    # lr_scheduler_params
    "sched": "cosine",
    "min_lr": 1e-4,
    "warmup_epochs": 5,
    "warmup_lr": 1e-6,
    "cooldown_epochs": 10,
    "t_max": 50,
    "cycle_momentum": False,
})

###############################################################################

if __name__ == "__main__":

    def auto_pos_int (x):
      val = int(x,0)
      if val <= 0:
          raise argparse.ArgumentTypeError("argument must be a positive int. Got {:d}.".format(val))
      return val
    
    parser = argparse.ArgumentParser(description='Run the shape embed pipeline')
    
    parser.add_argument(
        '-m', '--model', choices=models, metavar='MODEL'
      , help=f"The MODEL to use, one of {models} (default {params.model}).")
    parser.add_argument(
        '-d', '--dataset', nargs=2, metavar=('NAME', 'PATH')
      , help=f"The NAME of and PATH to the dataset (default: {params.dataset})")
    parser.add_argument(
        '-w', '--wandb-project', default="shape-embed", metavar='PROJECT'
      , help=f"The wandb PROJECT name")
    parser.add_argument(
        '-b', '--batch-size', metavar='BATCH_SIZE', type=auto_pos_int
      , help=f"The BATCH_SIZE for the run, a positive integer (default {params.batch_size})")
    parser.add_argument(
        '-l', '--latent-space-size', metavar='LATENT_SPACE_SIZE', type=auto_pos_int
      , help=f"The LATENT_SPACE_SIZE, a positive integer (default {params.latent_dim})")
    parser.add_argument(
        '-n', '--num-workers', metavar='NUM_WORKERS', type=auto_pos_int
      , help=f"The NUM_WORKERS for the run, a positive integer (default {params.num_workers})")
    parser.add_argument(
        '-e', '--num-epochs', metavar='NUM_EPOCHS', type=auto_pos_int
      , help=f"The NUM_EPOCHS for the run, a positive integer (default {params.epochs})")
    #parser.add_argument('--clear-checkpoints', action='store_true'
    #  , help='remove checkpoints')
    parser.add_argument('-v', '--verbose', action='count', default=0
      , help="Increase verbosity level by adding more \"v\".")
    
    # parse command line arguments
    clargs=parser.parse_args()
    
    # set verbosity level for vprint function
    vprint.lvl = clargs.verbose
    
    # update default params with clargs
    if clargs.model:
      params.model = clargs.model
    if clargs.dataset:
      params.dataset = clargs.dataset
    if clargs.wandb_project:
      params.wandb_project = clargs.wandb_project
    if clargs.batch_size:
      params.batch_size = clargs.batch_size
    if clargs.latent_space_size:
      interp_size = clargs.latent_space_size * 2
      params.input_dim = (params.input_dim[0], interp_size, interp_size)
      params.latent_dim = interp_size
      params.num_embeddings = interp_size
      params.num_hiddens = interp_size
    if clargs.num_workers:
      params.num_workers = clargs.num_workers
    if clargs.num_epochs:
      params.epochs = clargs.num_epochs
    
    # run main process
    main_process(params)