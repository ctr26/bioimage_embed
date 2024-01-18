# %%
import seaborn as sns
import pyefd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import make_scorer
import pandas as pd
from sklearn import metrics
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import umap
from torch.autograd import Variable
from types import SimpleNamespace
import numpy as np
import logging
from skimage import measure
import umap.plot
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch
from types import SimpleNamespace

# Deal with the filesystem
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from bioimage_embed import shapes
import bioimage_embed

# Note - you must have torchvision installed for this example

from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from bioimage_embed.lightning import DataModule

from torchvision import datasets
from bioimage_embed.shapes.transforms import (
    ImageToCoords,
    CropCentroidPipeline,
    DistogramToCoords,
    MaskToDistogramPipeline,
    RotateIndexingClockwise,
)

import matplotlib.pyplot as plt

from bioimage_embed.lightning import DataModule
import matplotlib as mpl
from matplotlib import rc

import logging
import pickle 
import base64
import hashlib

logger = logging.getLogger(__name__)

def hashing_fn(args):
    serialized_args = pickle.dumps(vars(args))
    hash_object = hashlib.sha256(serialized_args)
    hashed_string = base64.urlsafe_b64encode(hash_object.digest()).decode()
    return hashed_string

def scoring_df(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    # Define a dictionary of metrics
    scoring = {
        "accuracy": make_scorer(metrics.accuracy_score),
        "precision": make_scorer(metrics.precision_score, average="macro"),
        "recall": make_scorer(metrics.recall_score, average="macro"),
        "f1": make_scorer(metrics.f1_score, average="macro"),
    }

    # Create a random forest classifier
    clf = RandomForestClassifier()

    # Specify the number of folds
    k_folds = 10

    # Perform k-fold cross-validation
    cv_results = cross_validate(
        estimator=clf,
        X=X,
        y=y,
        cv=KFold(n_splits=k_folds),
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

    sns.set(style="white", context="notebook", rc={"figure.figsize": (width, height)})

    # matplotlib.use("TkAgg")
    interp_size = 128 * 2
    max_epochs = 100
    window_size = 128 * 2

    params = {
        "model":"resnet18_vqvae_legacy",
        "epochs": 75,
        "batch_size": 4,
        "num_workers": 2**4,
        "input_dim": (3, interp_size, interp_size),
        "latent_dim": interp_size,
        "num_embeddings": interp_size,
        "num_hiddens": interp_size,
        "num_residual_hiddens": 32,
        "num_residual_layers": 150,
        "pretrained": True,
        # "embedding_dim": 32,
        # "num_embeddings": 16,
        "commitment_cost": 0.25,
        "decay": 0.99,
        "frobenius_norm": False,
    }

    optimizer_params = {
        "opt": "LAMB",
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

    #dataset_path = "bbbc010/BBBC010_v1_foreground_eachworm"
    dataset_path = "shape_embed_data/data/bbbc010/BBBC010_v1_foreground_eachworm/"
    # dataset_path = "vampire/mefs/data/processed/Control"
    # dataset_path = "shape_embed_data/data/vampire/torchvision/Control/"
    # dataset_path = "vampire/torchvision/Control"
    # dataset = "bbbc010"

    # train_data_path = f"scripts/shapes/data/{dataset_path}"
    train_data_path = f"scripts/shapes/data/{dataset_path}"
    metadata = lambda x: f"results/{dataset_path}_{args.model}/{x}"

    path = Path(metadata(""))
    path.mkdir(parents=True, exist_ok=True)
    # %%

    transform_crop = CropCentroidPipeline(window_size)
    transform_dist = MaskToDistogramPipeline(
        window_size, interp_size, matrix_normalised=False
    )
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

    transform_mask_to_dist = transforms.Compose(
        [
            transform_mask_to_crop,
            transform_dist,
        ]
    )
    transform_mask_to_coords = transforms.Compose(
        [
            transform_mask_to_crop,
            transform_coords,
        ]
    )

    transforms_dict = {
        "none": transform_mask_to_gray,
        "transform_crop": transform_mask_to_crop,
        "transform_dist": transform_mask_to_dist,
        "transform_coords": transform_mask_to_coords,
    }

    train_data = {
        key: datasets.ImageFolder(train_data_path, transform=value)
        for key, value in transforms_dict.items()
    }

    for key, value in train_data.items():
        print(key, len(value))
        plt.imshow(train_data[key][0][0], cmap="gray")
        plt.imsave(metadata(f"{key}.png"), train_data[key][0][0], cmap="gray")
        # plt.show()
        plt.close()

    # plt.scatter(*train_data["transform_coords"][0][0])
    # plt.savefig(metadata(f"transform_coords.png"))
    # plt.show()

    # plt.imshow(train_data["transform_crop"][0][0], cmap="gray")
    # plt.scatter(*train_data["transform_coords"][0][0],c=np.arange(interp_size), cmap='rainbow', s=1)
    # plt.show()
    # plt.savefig(metadata(f"transform_coords.png"))

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
    plt.savefig(metadata(f"transform_coords.png"), bbox_inches="tight", pad_inches=0)

    # Close the plot
    plt.close()
    # import albumentations as A
    # %%
    gray2rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    transform = transforms.Compose(
        [
            transform_mask_to_dist,
            transforms.ToTensor(),
            RotateIndexingClockwise(p=1),
            gray2rgb,
        ]
    )

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
            print(f"Error occurred for image {idx}: {e}")

    # Create a Subset using the valid indices
    dataset = torch.utils.data.Subset(dataset, valid_indices)
    dataloader = DataModule(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # model = bioimage_embed.models.create_model("resnet18_vqvae_legacy", **vars(args))
    # 
    model = bioimage_embed.models.create_model(
        model=args.model,
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        pretrained=args.pretrained,
    )

    # model = bioimage_embed.models.factory.ModelFactory(**vars(args)).resnet50_vqvae_legacy()

    # lit_model = shapes.MaskEmbedLatentAugment(model, args)
    lit_model = shapes.MaskEmbed(model, args)
    test_data = dataset[0][0].unsqueeze(0)
    # test_lit_data = 2*(dataset[0][0].unsqueeze(0).repeat_interleave(3, dim=1),)
    test_output = lit_model.forward((test_data,))

    dataloader.setup()
    model.eval()

    model_dir = f"checkpoints/{hashing_fn(args)}"

    tb_logger = pl_loggers.TensorBoardLogger(f"logs/")
    wandb = pl_loggers.WandbLogger(project="bioimage-embed", name="shapes")

    Path(f"{model_dir}/").mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=f"{model_dir}/", save_last=True)
    wandb.watch(lit_model, log="all")

    trainer = pl.Trainer(
        logger=[wandb,tb_logger],
        gradient_clip_val=0.5,
        enable_checkpointing=True,
        devices=1,
        accelerator="gpu",
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback],
        min_epochs=50,
        max_epochs=args.epochs,
        log_every_n_steps=1,
    )
    # %%
    try:
        trainer.fit(
            lit_model, datamodule=dataloader, ckpt_path=f"{model_dir}/last.ckpt"
        )
    except:
        trainer.fit(lit_model, datamodule=dataloader)

    lit_model.eval()

    validation = trainer.validate(lit_model, datamodule=dataloader)
    # testing = trainer.test(lit_model, datamodule=dataloader)
    example_input = Variable(torch.rand(1, *args.input_dim))

    # torch.jit.save(lit_model.to_torchscript(), f"{model_dir}/model.pt")
    # torch.onnx.export(lit_model, example_input, f"{model_dir}/model.onnx")

    # %%
    # Inference

    dataloader = DataModule(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        # Transform is commented here to avoid augmentations in real data
        # HOWEVER, applying a the transform multiple times and averaging the results might produce better latent embeddings
        # transform=transform,
        # transform=transform,
    )
    dataloader.setup()

    predictions = trainer.predict(lit_model, datamodule=dataloader)

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
    df["Class"] = y
    # Map numeric classes to their labels
    idx_to_class = {0: "alive", 1: "dead"}
    df["Class"] = df["Class"].map(idx_to_class)
    df["Scale"] = scalings[:, 0].squeeze()
    df = df.set_index("Class")
    df_shape_embed = df.copy()

    ax = sns.relplot(
        data=df,
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
    plt.savefig(metadata(f"umap_no_axes.pdf"))
    # plt.show()
    plt.close()

    # %%

    X = df_shape_embed.to_numpy()
    y = df_shape_embed.index.values

    properties = [
        "area",
        "perimeter",
        "centroid",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
    ]
    dfs = []
    for i, data in enumerate(train_data["transform_crop"]):
        X, y = data
        # Do regionprops here
        # Calculate shape summary statistics using regionprops
        # We're considering that the mask has only one object, thus we take the first element [0]
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

    # Assuming 'dataset_contour' is your DataLoader for the dataset
    dfs = []
    for i, data in enumerate(train_data["transform_coords"]):
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
        print(trial["score_df"])
        trial["score_df"].to_csv(metadata(f"{trial['name']}_score_df.csv"))
        trial_df = pd.concat([trial_df, trial["score_df"]])
    trial_df = trial_df.drop(["fit_time", "score_time"], axis=1)

    trial_df.to_csv(metadata(f"trial_df.csv"))
    trial_df.groupby("trial").mean().to_csv(metadata(f"trial_df_mean.csv"))
    trial_df.plot(kind="bar")

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
    plt.savefig(metadata(f"trials_barplot.pdf"))
    plt.close()

    avs = (
        melted_df.set_index(["trial", "Metric"])
        .xs("test_f1", level="Metric", drop_level=False)
        .groupby("trial")
        .mean()
    )
    print(avs)
    # tikzplotlib.save(metadata(f"trials_barplot.tikz"))


if __name__ == "__main__":
    shape_embed_process()
