# %%
import seaborn as sns
import pyefd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_validate,
    KFold,
    train_test_split,
)
from sklearn.metrics import make_scorer
import pandas as pd
from sklearn import metrics
import matplotlib as mpl
from pathlib import Path
from sklearn.pipeline import Pipeline
from torch.autograd import Variable
from types import SimpleNamespace
import numpy as np
from skimage import measure
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch
from types import SimpleNamespace
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from umap import UMAP
# Deal with the filesystem
import torch.multiprocessing
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

torch.multiprocessing.set_sharing_strategy("file_system")

from bioimage_embed import shapes
import bioimage_embed
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from bioimage_embed.lightning import DataModule

from torchvision import datasets
from bioimage_embed.shapes.transforms import (
    ImageToCoords,
    CropCentroidPipeline,
    DistogramToCoords,
    RotateIndexingClockwise,
    CoordsToDistogram,
    AsymmetricDistogramToCoordsPipeline,
)
import matplotlib.pyplot as plt

from matplotlib import rc

import pickle
import base64
import hashlib

logger = logging.getLogger(__name__)

# Seed everything
np.random.seed(42)
pl.seed_everything(42)


def hashing_fn(args):
    serialized_args = pickle.dumps(vars(args))
    hash_object = hashlib.sha256(serialized_args)
    hashed_string = base64.urlsafe_b64encode(hash_object.digest()).decode()
    return hashed_string


def umap_plot(df, metadata, width=3.45, height=3.45 / 1.618):
    umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    mask = np.random.rand(len(df)) < 0.7

    semi_labels = df["Class"].copy()
    semi_labels[~mask] = -1  # Assuming -1 indicates unknown label for semi-supervision

    umap_embedding = umap_reducer.fit_transform(df, y=semi_labels)

    ax = sns.relplot(
        data=pd.DataFrame(umap_embedding, columns=["umap0", "umap1"]),
        x="umap0",
        y="umap1",
        hue="Class",
        palette="deep",
        alpha=0.5,
        edgecolor=None,
        s=5,
        height=height,
        aspect=0.5 * width / height,
    )

    sns.move_legend(
        ax,
        "upper center",
    )
    ax.set(xlabel=None, ylabel=None)
    sns.despine(left=True, bottom=True)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(metadata("umap_no_axes.pdf"))
    # plt.show()
    plt.close()


def scoring_df(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    # Define a dictionary of metrics
    scoring = {
        "accuracy": make_scorer(metrics.balanced_accuracy_score),
        "precision": make_scorer(metrics.precision_score, average="macro"),
        "recall": make_scorer(metrics.recall_score, average="macro"),
        "f1": make_scorer(metrics.f1_score, average="macro"),
        "roc_auc": make_scorer(metrics.roc_auc_score, average="macro"),
    }

    # Create a random forest classifier
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            #  ("pca", PCA(n_components=0.95, whiten=True, random_state=42)),
            ("clf", RandomForestClassifier()),
            # ("clf", DummyClassifier()),
        ]
    )

    # Specify the number of folds
    k_folds = 5

    # Perform k-fold cross-validation
    cv_results = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=StratifiedKFold(n_splits=k_folds),
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    # Put the results into a DataFrame
    return pd.DataFrame(cv_results)


def shape_embed_process():
    # Setting the font size
    mpl.rcParams["font.size"] = 10

    # rc("text", usetex=True)
    rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})
    width = 3.45
    height = width / 1.618
    plt.rcParams["figure.figsize"] = [width, height]

    sns.set(
        style="white",
        context="notebook",
        rc={"figure.figsize": (width, height)},
    )

    # matplotlib.use("TkAgg")
    interp_size = 128 * 2
    max_epochs = 100
    window_size = 128 * 2

    params = {
        "model": "resnet50_vqvae",
        "epochs": 250,
        "batch_size": 4,
        "num_workers": 2**4,
        "input_dim": (3, interp_size, interp_size),
        "latent_dim": interp_size,
        "num_embeddings": interp_size,
        "num_hiddens": interp_size,
        "pretrained": True,
        "commitment_cost": 0.25,
        "decay": 0.99,
        "frobenius_norm": False,
        # dataset = "bbbc010/BBBC010_v1_foreground_eachworm"
        # dataset = "vampire/mefs/data/processed/Control"
        "dataset": "synthcellshapes_dataset",
    }

    optimizer_params = {
        "opt": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "momentum": 0.9,
    }

    lr_scheduler_params = {
        "sched": "cosine",
        "min_lr": 1e-4,
        "warmup_epochs": 5,
        "warmup_lr": 1e-6,
        "cooldown_epochs": 10,
        "t_max": 50,
        "cycle_momentum": False,
    }

    args = SimpleNamespace(**params, **optimizer_params, **lr_scheduler_params)

    dataset_path = args.dataset

    # train_data_path = f"scripts/shapes/data/{dataset_path}"
    train_data_path = f"data/{dataset_path}"
    metadata = lambda x: f"results/{dataset_path}_{args.model}/{x}"

    path = Path(metadata(""))
    path.mkdir(parents=True, exist_ok=True)
    # %%

    transform_crop = CropCentroidPipeline(window_size)
    # transform_dist = MaskToDistogramPipeline(
    # window_size, interp_size, matrix_normalised=False
    # )
    transform_coord_to_dist = CoordsToDistogram(interp_size, matrix_normalised=False)
    transform_mdscoords = DistogramToCoords(window_size)
    transform_coords = ImageToCoords(window_size)

    transform_mask_to_gray = transforms.Compose([transforms.Grayscale(1)])

    transform_mask_to_crop = transforms.Compose(
        [
            # transforms.ToTensor(),
            transform_mask_to_gray,
            transform_crop,
        ]
    )

    transform_mask_to_coords = transforms.Compose(
        [
            transform_mask_to_crop,
            transform_coords,
        ]
    )

    transform_mask_to_dist = transforms.Compose(
        [
            transform_mask_to_coords,
            transform_coord_to_dist,
        ]
    )

    gray2rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    transform = transforms.Compose(
        [
            transform_mask_to_dist,
            transforms.ToTensor(),
            RotateIndexingClockwise(p=1),
            gray2rgb,
        ]
    )

    transforms_dict = {
        "none": transform_mask_to_gray,
        "transform_crop": transform_mask_to_crop,
        "transform_dist": transform_mask_to_dist,
        "transform_coords": transform_mask_to_coords,
    }

    # Apply transform to find which images don't work
    dataset = datasets.ImageFolder(train_data_path, transform=transform)

    valid_indices = []
    # Iterate through the dataset and apply the transform to each image
    for idx in range(len(dataset)):
        try:
            image, label = dataset[idx]
            # If the transform works without errors, add the index to the list of valid indices
            valid_indices.append(idx)
        except Exception as e:
            # A better way to do with would be with batch collation
            logger.warning(f"Error occurred for image {idx}: {e}")

    train_data = {
        key: torch.utils.data.Subset(
            datasets.ImageFolder(train_data_path, transform=value),
            valid_indices,
        )
        for key, value in transforms_dict.items()
    }

    dataset = torch.utils.data.Subset(
        datasets.ImageFolder(train_data_path, transform=transform),
        valid_indices,
    )

    for key, value in train_data.items():
        logger.info(key, len(value))
        plt.imshow(np.array(train_data[key][0][0]), cmap="gray")
        plt.imsave(metadata(f"{key}.png"), train_data[key][0][0], cmap="gray")
        # plt.show()
        plt.close()

    # Retrieve the coordinates and cropped image
    coords = train_data["transform_coords"][0][0]
    crop_image = train_data["transform_crop"][0][0]

    fig = plt.figure(frameon=True)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Display the cropped image using grayscale colormap
    plt.imshow(crop_image, cmap="gray_r")

    # Scatter plot with smaller point size
    plt.scatter(*coords, c=np.arange(interp_size), cmap="rainbow", s=2)

    # Save the plot as an image without border and coordinate axes
    plt.savefig(metadata("transform_coords.png"), bbox_inches="tight", pad_inches=0)

    # Close the plot
    plt.close()

    # Create a Subset using the valid indices
    dataloader = DataModule(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = bioimage_embed.models.create_model(
        model=args.model,
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        pretrained=args.pretrained,
    )

    # lit_model = shapes.MaskEmbedLatentAugment(model, args)
    lit_model = shapes.MaskEmbed(model, args)
    test_data = dataset[0][0].unsqueeze(0)
    # test_lit_data = 2*(dataset[0][0].unsqueeze(0).repeat_interleave(3, dim=1),)
    test_output = lit_model.forward((test_data,))

    dataloader.setup()
    model.eval()

    model_dir = f"checkpoints/{hashing_fn(args)}"

    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    wandb = pl_loggers.WandbLogger(project="bioimage-embed", name="shapes")

    Path(f"{model_dir}/").mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{model_dir}/",
        save_last=True,
        save_top_k=1,
        monitor="loss/val",
        mode="min",
    )
    wandb.watch(lit_model, log="all")

    trainer = pl.Trainer(
        logger=[wandb, tb_logger],
        gradient_clip_val=0.5,
        enable_checkpointing=True,
        devices=1,
        accelerator="gpu",
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback],
        min_epochs=50,
        max_epochs=args.epochs,
        # callbacks=[EarlyStopping(monitor="loss/val", mode="min")],
        log_every_n_steps=1,
    )
    # %%

    # Determine the checkpoint path for resuming
    last_checkpoint_path = f"{model_dir}/last.ckpt"
    best_checkpoint_path = checkpoint_callback.best_model_path

    # Check if a last checkpoint exists to resume from
    if os.path.isfile(last_checkpoint_path):
        resume_checkpoint = last_checkpoint_path
    elif best_checkpoint_path and os.path.isfile(best_checkpoint_path):
        resume_checkpoint = best_checkpoint_path
    else:
        resume_checkpoint = None

    trainer.fit(lit_model, datamodule=dataloader, ckpt_path=resume_checkpoint)

    lit_model.eval()

    validation = trainer.validate(lit_model, datamodule=dataloader)
    # testing = trainer.test(lit_model, datamodule=dataloader)
    example_input = Variable(torch.rand(1, *args.input_dim))

    # torch.jit.save(lit_model.to_torchscript(), f"{model_dir}/model.pt")
    # torch.onnx.export(lit_model, example_input, f"{model_dir}/model.onnx")

    # %%
    # Inference on full dataset
    dataloader = DataModule(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        # Transform is commented here to avoid augmentations in real data
        # HOWEVER, applying the transform multiple times and averaging the results might produce better latent embeddings
        # transform=transform,
    )
    dataloader.setup()

    predictions = trainer.predict(lit_model, datamodule=dataloader)

    test_dist_pred = predictions[0].out.recon_x
    plt.imsave(metadata("test_dist_pred.png"), test_dist_pred.mean(axis=(0, 1)))
    plt.close()

    test_dist_in = predictions[0].x.data
    plt.imsave(metadata("test_dist_in.png"), test_dist_in.mean(axis=(0, 1)))
    plt.close()

    test_pred_coords = AsymmetricDistogramToCoordsPipeline(window_size=window_size)(
        np.array(test_dist_pred[:, 0, :, :].unsqueeze(dim=0))
    )

    plt.scatter(*test_pred_coords[0, 0].T)
    # Save the plot as an image without border and coordinate axes
    plt.savefig(metadata("test_pred_coords.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

    # Use the namespace variables
    latent_space = torch.stack([d.out.z.flatten() for d in predictions])
    scalings = torch.stack([d.x.scalings.flatten() for d in predictions])
    idx_to_class = {v: k for k, v in dataset.dataset.class_to_idx.items()}
    y = np.array([int(data[-1]) for data in dataloader.predict_dataloader()])

    y_partial = y.copy()
    indices = np.random.choice(y.size, int(0.3 * y.size), replace=False)
    y_partial[indices] = -1
    y_blind = -1 * np.ones_like(y)

    df = pd.DataFrame(latent_space.numpy())
    df["Class"] = pd.Series(y).map(idx_to_class).astype("category")
    df["Scale"] = scalings[:, 0].squeeze()
    df = df.set_index("Class")
    df_shape_embed = df.copy()

    # %%

    X = df_shape_embed.to_numpy()
    y = df_shape_embed.index

    properties = [
        "area",
        "perimeter",
        "centroid",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
    ]
    dfs = []
    # Distance matrix data
    for i, data in enumerate(tqdm(train_data["transform_crop"])):
        X, y = data
        # Do regionprops here
        # Calculate shape summary statistics using regionprops
        # We're considering that the mask has only one object, so we take the first element [0]
        # props = regionprops(np.array(X).astype(int))[0]
        props_table = measure.regionprops_table(
            np.array(X).astype(int), properties=properties
        )

        # Store shape properties in a dataframe
        df = pd.DataFrame(props_table)

        # Assuming the class or label is contained in 'y' variable
        df["class"] = y
        df.set_index("class", inplace=True)
        dfs.append(df)

    df_regionprops = pd.concat(dfs)

    dfs = []
    for i, data in enumerate(tqdm(train_data["transform_coords"])):
        # Convert the tensor to a numpy array
        X, y = data

        # Feed it to PyEFD's calculate_efd function
        coeffs = pyefd.elliptic_fourier_descriptors(X, order=10, normalize=False)
        # coeffs_df = pd.DataFrame({'class': [y], 'norm_coeffs': [norm_coeffs.flatten().tolist()]})

        norm_coeffs = pyefd.normalize_efd(coeffs)
        df = pd.DataFrame(
            {
                "norm_coeffs": norm_coeffs.flatten().tolist(),
                "coeffs": coeffs.flatten().tolist(),
            }
        ).T.rename_axis("coeffs")
        df["class"] = y
        df.set_index("class", inplace=True, append=True)
        dfs.append(df)

    df_pyefd = pd.concat(dfs)

    trials = [
        {
            "name": "mask_embed",
            "features": df_shape_embed.to_numpy(),
            "labels": df_shape_embed.index,
        },
        {
            "name": "fourier_coeffs",
            "features": df_pyefd.xs("coeffs", level="coeffs"),
            "labels": df_pyefd.xs("coeffs", level="coeffs").index,
        },
        # {"name": "fourier_norm_coeffs",
        #  "features": df_pyefd.xs("norm_coeffs", level="coeffs"),
        #  "labels": df_pyefd.xs("norm_coeffs", level="coeffs").index
        # }
        {
            "name": "regionprops",
            "features": df_regionprops,
            "labels": df_regionprops.index,
        },
    ]

    trial_df = pd.DataFrame()
    for trial in trials:
        X = trial["features"]
        y = trial["labels"]
        trial["score_df"] = scoring_df(X, y)
        trial["score_df"]["trial"] = trial["name"]
        logger.info(trial["score_df"])
        trial["score_df"].to_csv(metadata(f"{trial['name']}_score_df.csv"))
        trial_df = pd.concat([trial_df, trial["score_df"]])
    trial_df = trial_df.drop(["fit_time", "score_time"], axis=1)

    trial_df.to_csv(metadata("trial_df.csv"))
    trial_df.groupby("trial").mean().to_csv(metadata("trial_df_mean.csv"))
    trial_df.plot(kind="bar")

    avg = trial_df.groupby("trial").mean()
    logger.info(avg)
    avg.to_latex(metadata("trial_df.tex"))

    melted_df = trial_df.melt(id_vars="trial", var_name="Metric", value_name="Score")
    # fig, ax = plt.subplots(figsize=(width, height))
    ax = sns.catplot(
        data=melted_df,
        kind="bar",
        x="trial",
        hue="Metric",
        y="Score",
        errorbar="se",
        height=height,
        aspect=width * 2**0.5 / height,
    )
    # ax.xtick_params(labelrotation=45)
    # plt.legend(loc='lower center', bbox_to_anchor=(1, 1))
    # sns.move_legend(ax, "lower center", bbox_to_anchor=(1, 1))
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    plt.savefig(metadata("trials_barplot.pdf"))
    plt.close()

    avs = (
        melted_df.set_index(["trial", "Metric"])
        .xs("test_f1", level="Metric", drop_level=False)
        .groupby("trial")
        .mean()
    )
    logger.info(avs)
    # tikzplotlib.save(metadata(f"trials_barplot.tikz"))


if __name__ == "__main__":
    shape_embed_process()
