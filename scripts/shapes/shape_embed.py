# %%
import seaborn as sns
import pyefd
import tikzplotlib
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
import tikzplotlib

import umap.plot
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch

# Deal with the filesystem
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
)

# from bioimage_embed.models import Mask_VAE, VQ_VAE, VAE
import matplotlib.pyplot as plt

from bioimage_embed.lightning import DataModule
import matplotlib as mpl
from matplotlib import rc

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
        "epochs": 75,
        "batch_size": 4,
        "num_workers": 2**4,
        # "window_size": 64*2,
        "num_workers": 1,
        "input_dim": (1, window_size, window_size),
        # "channels": 3,
        "latent_dim": 16,
        "num_embeddings": 16,
        "num_hiddens": 16,
        "num_residual_hiddens": 32,
        "num_residual_layers": 150,
        # "embedding_dim": 32,
        # "num_embeddings": 16,
        "commitment_cost": 0.25,
        "decay": 0.99,
        "loss_weights": [1, 1, 1, 1],
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

    # channels = 3


    # input_dim = (params["channels"], params["window_size"], params["window_size"])
    args = SimpleNamespace(**params, **optimizer_params, **lr_scheduler_params)

    dataset_path = "bbbc010"
    #dataset_path = "vampire/mefs/data/processed/Control"
    #dataset_path = "bbbc010/BBBC010_v1_foreground_eachworm"
    # dataset_path = "vampire/torchvision/Control"
    # dataset = "bbbc010"
    model_name = "vqvae"

    #train_data_path = f"scripts/shapes/data/{dataset_path}"
    train_data_path = f"data/{dataset_path}"
    metadata = lambda x: f"results/{dataset_path}_{model_name}/{x}"

    path = Path(metadata(""))
    path.mkdir(parents=True, exist_ok=True)
    model_dir = f"models/{dataset_path}_{model_name}"
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

    # train_data = torchvision.datasets.ImageFolder(
    # "/home/ctr26/gdrive/+projects/idr_autoencode_torch/data/bbbc010"
    # )
    # train_dataset_crop = DatasetGlob(
    #     train_dataset_glob, transform=CropCentroidPipeline(window_size))
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

    # img_squeeze = train_dataset[0].unsqueeze(0)
    # %%


    # def my_collate(batch):
    #     batch = list(filter(lambda x: x is not None, batch))
    #     return torch.utils.data.dataloader.default_collate(batch)

    transform = transforms.Compose([transform_mask_to_dist, transforms.ToTensor()])

    dataset = datasets.ImageFolder(train_data_path, transform=transform)

    valid_indices = []
    # Iterate through the dataset and apply the transform to each image
    for idx in range(len(dataset)):
        try:
            image, label = dataset[idx]
            # If the transform works without errors, add the index to the list of valid indices
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error occurred for image {idx}: {e}")

    # Create a Subset using the valid indices
    dataset = torch.utils.data.Subset(dataset, valid_indices)
    dataloader = DataModule(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # transform=transform,
    )


    # dataloader = DataLoader(train_dataset, batch_size=batch_size,
    #                         shuffle=True, num_workers=2**4, pin_memory=True, collate_fn=my_collate)

    model = bioimage_embed.models.create_model("resnet18_vqvae_legacy", **vars(args))

    lit_model = shapes.MaskEmbedLatentAugment(model, args)
    lit_model = shapes.MaskEmbed(model, args)

    # model = Mask_VAE("VAE", 1, 64,
    #                      #  hidden_dims=[32, 64],
    #                      image_dims=(interp_size, interp_size))

    # model = Mask_VAE(VAE())
    # %%
    # lit_model = LitAutoEncoderTorch(model)

    dataloader.setup()
    model.eval()
    # %%


    # model_name = model._get_name()
    model_dir = f"my_models/{dataset_path}_{model._get_name()}_{lit_model._get_name()}"

    tb_logger = pl_loggers.TensorBoardLogger(f"logs/")

    Path(f"{model_dir}/").mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=f"{model_dir}/", save_last=True)

    trainer = pl.Trainer(
        logger=tb_logger,
        gradient_clip_val=0.5,
        enable_checkpointing=True,
        devices=1,
        accelerator="gpu",
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback],
        min_epochs=50,
        max_epochs=args.epochs,
    )  # .from_argparse_args(args)

    # %%

    try:
        trainer.fit(lit_model, datamodule=dataloader, ckpt_path=f"{model_dir}/last.ckpt")
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
    # predict_dataloader = DataLoader(dataset, batch_size=1)


    dataloader = DataModule(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        # transform=transform,
    )
    dataloader.setup()

    predictions = trainer.predict(lit_model, datamodule=dataloader)
    latent_space = torch.stack(
        [prediction.z.flatten() for prediction in predictions[:-1]], dim=0
    )

    idx_to_class = {v: k for k, v in dataset.dataset.class_to_idx.items()}


    y = np.array([int(data[-1]) for data in dataloader.predict_dataloader()])[:-1]

    y_partial = y.copy()
    indices = np.random.choice(y.size, int(0.3 * y.size), replace=False)

    y_partial[indices] = -1

    classes = np.array([idx_to_class[i] for i in y])


    # y = torch.stack([data[-1] for data in dataloader.dataset[:-1], dim=0)
    # y = torch.stack([prediction.y for prediction in predictions[:-1]], dim=0)
    # umap_space = umap.UMAP().fit(latent_space, y=y)
    # umap_space = umap.UMAP().fit_transform(latent_space.numpy(), y=y)
    mapper = umap.UMAP().fit(latent_space.numpy(), y=y)
    semi_supervised_latent = mapper.transform(latent_space.numpy())

    df = pd.DataFrame(semi_supervised_latent, columns=["umap0", "umap1"])
    df["Class"] = y
    # Map numeric classes to their labels
    idx_to_class = {0: "alive", 1: "dead"}
    df["Class"] = df["Class"].map(idx_to_class)


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

    # ax.annotate('', xy=(0, 0.2), xytext=(0, 0), xycoords='axes fraction',
    #              arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5))
    # ax.annotate('', xy=(0.2*height/width, 0), xytext=(0, 0), xycoords='axes fraction',  arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5))

    sns.move_legend(
        ax,
        "upper center",
    )
    ax.set(xlabel=None, ylabel=None)
    sns.despine(left=True, bottom=True)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    # plt.legend(loc="upper center", ncol=2,bbox_to_anchor=(0.5, 1.5))
    plt.tight_layout()
    # umap.plot.points(mapper, labels=classes,width=width*dpi, height=height*dpi)
    # plt.savefig(metadata(f"umap.png"))
    plt.savefig(metadata(f"umap_no_axes.pdf"))
    # plt.show()
    plt.close()


    # %%


    X = latent_space.numpy()
    y = classes


    def scoring_df(X, y):
        # X = semi_supervised_latent
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


    # mask_embed_score_df = scoring_df(X, y)
    # Print the DataFrame
    # print(mask_embed_score_df)
    # mask_embed_score_df.to_csv(metadata(f"mask_embed_score_df.csv"))
    from skimage import measure

    dfs = []
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
        {"name": "mask_embed", "features": latent_space.numpy(), "labels": classes},
        {
            "name": "fourier_coeffs",
            "features": df_pyefd.xs("coeffs", level="coeffs"),
            "labels": df_pyefd.xs("coeffs", level="coeffs").index,
        },
        # {"name": "fourier_norm_coeffs",
        #  "features": df_pyefd.xs("norm_coeffs", level="coeffs"),
        #  "labels": df_pyefd.xs("norm_coeffs", level="coeffs").index
        # }
        {"name": "regionprops", "features": df_regionprops, "labels": df_regionprops.index},
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
