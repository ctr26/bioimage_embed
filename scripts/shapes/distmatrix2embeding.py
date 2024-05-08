from torchvision import datasets, transforms
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import umap
import umap.plot
import bokeh.plotting
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import bioimage_embed
import bioimage_embed.shapes
import bioimage_embed.lightning
from bioimage_embed.lightning import DataModule
from pytorch_lightning import loggers as pl_loggers
import argparse
import datetime
import pathlib
import torch
import types
import re

# Seed everything
np.random.seed(42)
pl.seed_everything(42)

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
        #lambda x: x*1000, # scale the matrix
        #lambda x: x / x.max(), # normalize each element to one using the max value (0-1)
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

    extra_params = {}
    if re.match(".*_beta_vae", params.model):
      extra_params['beta'] = params.model_beta_vae_beta
    model = bioimage_embed.models.create_model(
        model=params.model,
        input_dim=params.input_dim,
        latent_dim=params.latent_dim,
        pretrained=params.pretrained,
        **extra_params
    )
    lit_model = bioimage_embed.shapes.MaskEmbed(model, params)
    vprint(1, f'model ready')

    # WandB logger
    ###########################################################################
    jobname = f"{params.model}_{'_'.join([f'{k}{v}' for k, v in extra_params.items()])}_{params.latent_dim}_{params.batch_size}_{params.dataset[0]}"
    wandblogger = pl_loggers.WandbLogger(entity=params.wandb_entity, project=params.wandb_project, name=jobname)
    wandblogger.watch(lit_model, log="all")
    # TODO: Sanity check:
    # test_data = dataset[0][0].unsqueeze(0)
    # test_output = lit_model.forward((test_data,))

    # Train the model
    ###########################################################################
    
    trainer = pl.Trainer(
        logger=[wandblogger],
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
    vprint(1, f'Validate the model')
    validation = trainer.validate(lit_model, datamodule=dataloader)
    
    #TODO: Test the model
    ###########################################################################  
    vprint(1, f'Test the model')
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
    
    # Predict
    ###########################################################################
    predictions = trainer.predict(lit_model, datamodule=dataloader)
    filenames = [sample[0] for sample in dataloader.get_dataset().samples]
    class_indices = np.array([int(data[-1]) for data in dataloader.predict_dataloader()])
    
    #TODO: Pull the embedings and reconstructed distance matrices
    ###########################################################################
    # create the output directory
    output_dir = params.output_dir
    if output_dir is None:
      output_dir = f'./{params.model}_{params.latent_dim}_{params.batch_size}_{params.dataset[0]}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    for class_label in dataset.classes:
      pathlib.Path(f'{output_dir}/{class_label}').mkdir(parents=True, exist_ok=True)
    # Save the latent space
    vprint(1, f'pull the embedings')
    latent_space = torch.stack([d.out.z.flatten() for d in predictions]).numpy()
    np.save(f'{output_dir}/latent_space.npy', latent_space)
    df = pd.DataFrame(latent_space)
    df['class_idx'] = class_indices
    df['class'] = [dataset.classes[x] for x in class_indices]
    df['fname'] = filenames
    df.to_pickle(f'{output_dir}/latent_space.pkl')
    # Save the (original input and) reconstructions
    for i, (pred, class_idx, fname) in enumerate(zip(predictions, class_indices, filenames)):
      vprint(5, f'pred#={i}, class_idx={class_idx}, fname={fname}')
      class_label = dataset.classes[class_idx]
      np.save(f'{output_dir}/{class_label}/original_{i}_{class_label}.npy', pred.x.data[0,0])
      np.save(f'{output_dir}/{class_label}/reconstruction_{i}_{class_label}.npy', pred.out.recon_x[0,0])
    # umap
    vprint(4, f'generate umap')
    umap_model = umap.UMAP(n_neighbors=50, min_dist=0.8, n_components=2, random_state=42)
    mapper = umap_model.fit(df.drop(['class_idx','class','fname'], axis=1))
    umap.plot.points(mapper, labels=np.array(df['class']))
    plt.savefig(f'{output_dir}/umap.png')
    #p = umap.plot.interactive(mapper, labels=df['class_idx'], hover_data=df[['class','fname']])
    p = umap.plot.interactive(mapper, values=df.drop(['class_idx','class','fname'], axis=1).mean(axis=1), theme='viridis', hover_data=df[['class','fname']])
    # save interactive plot as html
    bokeh.plotting.output_file(f"{output_dir}/umap.html")
    bokeh.plotting.save(p)

    # kmean and clustering information
    # Perform KMeans clustering on the UMAP result
    vprint(4, f'cluster data with kmean')
    n_clusters = 4  # Define the number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    umap_result = umap_model.fit_transform(latent_space)
    cluster_labels = kmeans.fit_predict(umap_result)

    # Concatenate the original data, UMAP result, and cluster labels
    data_with_clusters = np.column_stack((latent_space, umap_result, cluster_labels))

    # Convert to DataFrame for better handling
    columns = [f'Feature_{i}' for i in range(latent_space.shape[1])] + \
              ['UMAP_Dimension_1', 'UMAP_Dimension_2', 'Cluster_Label']
    df = pd.DataFrame(data_with_clusters, columns=columns)
    df['fname'] = filenames

    df.to_csv(f'{output_dir}/clustered_data.csv', index=False)

    # Plot the UMAP result with cluster labels
    plt.figure(figsize=(10, 8))
    for i in range(n_clusters):
      plt.scatter(umap_result[cluster_labels == i, 0], umap_result[cluster_labels == i, 1], label=f'Cluster {i+1}', s=5)
    plt.title('UMAP Visualization of Latent Space with KMeans Clustering')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()

    # Save the figure
    plt.savefig(f'{output_dir}/umap_with_kmeans_clusters.png')
    
    # Test embeding for a classifcation task
    

# default parameters
###############################################################################

models = [
  "resnet18_vae"
, "resnet50_vae"
, "resnet18_beta_vae"
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
    "dataset": ("tiny_dist", "/nfs/research/uhlmann/afoix/distmat_datasets/tiny_synthcellshapes_dataset_distmat"),
    # model-specific params
    "model_beta_vae_beta": 1,
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
        '--model-beta-vae-beta', type=float, metavar='BETA'
      , help=f"The BETA parameter to use for a beta-vae model.")
    parser.add_argument(
        '-d', '--dataset', nargs=2, metavar=('NAME', 'PATH')
      , help=f"The NAME of and PATH to the dataset (default: {params.dataset})")
    parser.add_argument(
        '-o', '--output-dir', metavar='OUTPUT_DIR', default=None
      , help=f"The OUTPUT_DIR path to use to dump results")
    parser.add_argument(
        '--wandb-entity', default="foix", metavar='WANDB_ENTITY'
      , help=f"The WANDB_ENTITY name")
    parser.add_argument(
        '--wandb-project', default="simply-shape", metavar='WANDB_PROJECT'
      , help=f"The WANDB_PROJECT name")
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
    if clargs.model_beta_vae_beta:
      params.model_beta_vae_beta = clargs.model_beta_vae_beta
    params.output_dir = clargs.output_dir
    if clargs.dataset:
      params.dataset = clargs.dataset
    if clargs.wandb_entity:
      params.wandb_entity = clargs.wandb_entity
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
