from torchvision import datasets, transforms
import pytorch_lightning as pl
import bioimage_embed
import bioimage_embed.shapes
import bioimage_embed.lightning
import argparse
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

# Main process
###############################################################################

def main_process(params):

    # Loading the data (matrices)
    ###########################################################################

    preproc_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(params.dataset[1], transform = preproc_transform)
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
        enable_checkpointing=True,
        devices=1,
        #TODO accelerator="gpu",
        accumulate_grad_batches=4,
        #TODO callbacks=[checkpoint_callback],
        min_epochs=50,
        max_epochs=params.epochs,
        log_every_n_steps=1,
    )
    trainer.fit(lit_model, datamodule=dataloader)
    lit_model.eval()
    vprint(1, f'trainer fitted')

    # Pull the embedings
    ###########################################################################
    vprint(1, f'TODO')

# default parameters
###############################################################################

params = types.SimpleNamespace(**{
    # general params
    "model":"resnet18_vae",
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
    "dataset": "bbbc010/BBBC010_v1_foreground_eachworm",
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
    parser.add_argument(
        '-m', '--model', choices=models, default=models[0], metavar='MODEL'
      , help=f"The MODEL to use, one of {models} (default {models[0]}).")
    parser.add_argument(
        '-d', '--dataset', nargs=2, default=("vampire", "vampire/torchvision/Control/"), metavar=('NAME', 'PATH')
      , help=f"The NAME of and PATH to the dataset")
    parser.add_argument(
        '-w', '--wandb-project', default="shape-embed", metavar='PROJECT'
      , help=f"The wandb PROJECT name")
    parser.add_argument(
        '-b', '--batch-size', default=int(4), metavar='BATCH_SIZE', type=auto_pos_int
      , help="The BATCH_SIZE for the run, a positive integer (default 4)")
    parser.add_argument(
        '-l', '--latent-space-size', default=int(128), metavar='LATENT_SPACE_SIZE', type=auto_pos_int
      , help="The LATENT_SPACE_SIZE, a positive integer (default 128)")
    parser.add_argument(
        '-n', '--num-workers', default=int(2**4), metavar='NUM_WORKERS', type=auto_pos_int
      , help="The NUM_WORKERS for the run, a positive integer (default 2**4)")
    parser.add_argument(
        '-e', '--num-epochs', default=int(150), metavar='NUM_EPOCHS', type=auto_pos_int
      , help="The NUM_EPOCHS for the run, a positive integer (default 150)")
    #parser.add_argument('--clear-checkpoints', action='store_true'
    #  , help='remove checkpoints')
    parser.add_argument('-v', '--verbose', action='count', default=0
      , help="Increase verbosity level by adding more \"v\".")
    
    # parse command line arguments
    clargs=parser.parse_args()
    
    # set verbosity level for vprint function
    vprint.lvl = clargs.verbose
    
    # update default params with clargs
    params.model = clargs.model
    params.dataset = clargs.dataset
    params.wandb_project = clargs.wandb_project
    params.batch_size = clargs.batch_size
    interp_size = clargs.latent_space_size * 2
    params.input_dim = (3, interp_size, interp_size)
    params.latent_dim = interp_size
    params.num_embeddings = interp_size
    params.num_hiddens = interp_size
    params.num_workers = clargs.num_workers
    params.epochs = clargs.num_epochs
    
    # run main process
    main_process(params)