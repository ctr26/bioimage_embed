# %%
# import torchvision
# from omero.ObjectFactoryRegistrar import ExperimentObjectFactory
# %%
import torch
import numpy as np
import os

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

# from idr import connection
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

# %%
gpus = 0
# workers = 16
# batch_size = 16
# learning_rate = 1e-3
# epochs = 100
# root_dir = "/nfs/bioimage/drop/"
mode = "local"
training_dir = "data"

mode = "gcp"
training_dir = "idr-hipsci"
glob_pattern = "**/*.tiff"
# num_workers = os.cpu_count()
gcloud_secret_file = "gcloud_secret.json"
# accumulate_grad_batches
# gcloud_secret_file = "idr-hipsci/**/*.tiff"
# gcloud_project = "prj-ext-dev-bia-binder-113155"
# gcloud_glob = "gcloud_secret.json"


import argparse

parser = argparse.ArgumentParser()

# parser = argparse.ArgumentParser()
# # parser.add_argument("--gpus", type=int, default=gpus)
# # parser.add_argument("--workers", type=int, default=workers)
# # parser.add_argument("--batchsize", type=int, default=batch_size)
# # parser.add_argument(
# #     "--learning_rate", type=float, default=learning_rate
# # )  # Low learning rates converge better
# # parser.add_argument("--epochs", type=int, default=epochs)
parser.add_argument("--training_dir", type=str, default=training_dir)
parser.add_argument("--mode", type=str, default=mode)
# # parser.add_argument("--project", type=str, default=project)
parser.add_argument("--glob_pattern", type=str, default=glob_pattern)
# parser.add_argument("--num_workers", type=int, default=num_workers)
parser.add_argument("--gcloud_secret_file", type=str, default=gcloud_secret_file)


# parser.add_argument("--gpus", type=int, default=gpus)

# parser.add_argument("--gcloud_glob", type=str, default="idr-hipsci/**/*.tiff")
# parser.add_argument("--gcloud_project", type=str, default="prj-ext-dev-bia-binder-113155")
# parser.add_argument("--gcloud_secret_file", type=str, default="gcloud_secret.json")

parser = pl.Trainer.add_argparse_args(parser)

# %TODO add model params here
try:
    args, unknown = parser.parse_known_args()
except:
    args = parser.parse_args([])
print(vars(args))
print(unknown)

globals().update(vars(args))
# args_dict = vars(parser.parse_args())


# try:
#     args = parser.parse_known_args([
#                     "--training_dir",
#                     "--mode",
#                     "--glob_pattern"])
# # globals().update(vars(args))
# except:
#     args = parser.parse_known_args([])
# globals().update(vars(args))
# print(vars(args))

# print(args_dict)


# %%
# gcs = GCSFileSystem(...)
# storage_options

# ACCESS_KEY = "xx"
# SECRET_KEY = "yy"
# boto3.set_stream_logger('')

# session = Session(aws_access_key_id=ACCESS_KEY,
#                   aws_secret_access_key=SECRET_KEY,
#                   region_name="EU-WEST2")

# session.events.unregister('before-parameter-build.s3.ListObjects',
#                           set_list_objects_encoding_type_url)

# s3 = session.resource('s3', endpoint_url='https://storage.googleapis.com',
#                       config=Config(signature_version='s3v4'))


# bucket = s3.Bucket('idr-hipsci')

# for f in bucket.objects.all():
#         print(f.key)
# %%


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


# import NPC8nodes
from torch import nn

# from scipy.ndimage import convolve
# import seaborn as sns
from torch.utils.data import Dataset

cuda_availability = torch.cuda.is_available()
if cuda_availability:
    device = torch.device("cuda:{}".format(torch.cuda.current_device()))
    print(f"{torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
print(f"CUDA: {cuda_availability}")
# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# https://pytorch.org/vision/stable/auto_examples/plot_scripted_tensor_transforms.html#sphx-glr-auto-examples-plot-scripted-tensor-transforms-py
# https://amirakbarnejad.github.io/Tutorial/tutorial_section1.html
# %%
# index_num = 4684936
# conn = connection("idr.openmicroscopy.org",verbose=0)
#  %%
# file_list_path = os.path.join(root_dir,"tiff_list.txt")
# file_list = np.loadtxt(file_list_path)


# file_list

# gcs = None
#  %%

#  %%
# from boto3.session import Session
# from botocore.client import Config
# from botocore.handlers import set_list_objects_encoding_type_url

# import boto3

import json

with open(gcloud_secret_file) as json_file:
    tokengcs = json.load(json_file)

# from dask_image import imread
# # x = imread.imread('gs://idr-hipsci/*.tif',storage_options={'token':tokengcs})
import gcsfs

# cat_file
# import matplotlib.pyplot as plt
# import imageio
# plt.imshow(read_tif_from_gcs(gcs,file))

#  %%
# gcs = gcsfs.GCSFileSystem(project=gcloud_project,token=tokengcs)

# print(file_list[0])
# %%
import io

# file = file_list[0];file


# %%

# import imageio
# imageio.core.asarray(imageio.imread(current_image, "TIFF"))
# %%


# %%
class IDRDataSet(Dataset):
    def __init__(
        self,
        transform=None,
        training_dir="",
        mode="local",
        glob_pattern="**/*.tiff",
    ):
        super(IDRDataSet).__init__()
        self.transform = transform
        self.training_dir = training_dir
        self.glob_pattern = glob_pattern
        self.mode = mode
        # if training_dir == "None":
        # self.mode = "http"
        if self.mode == "local":
            self.file_list = glob(
                os.path.join(training_dir, glob_pattern), recursive=True
            )
        if self.mode == "gcp":
            # gcs = gcsfs.GCSFileSystem(project=gcloud_project,token=tokengcs)
            gcs = gcsfs.GCSFileSystem(
                token=tokengcs, access="read_only", skip_instance_cache=True
            )
            self.gcs_token = tokengcs
            self.file_list = gcs.glob(os.path.join(training_dir, glob_pattern))

        # assert end > start, "this example code only works with end >= start"
        # self.start = start
        # self.end = end
        # self.conn = connection("idr.openmicroscopy.org")
        # self.loader = default_loader
        # self.conn = connection("idr.openmicroscopy.org")

    def __getitem__(self, index):
        # conn = self.conn
        self.index = index

        if self.mode == "http":
            image = self.get_idr_image(index)
        # if (self.mode=="cloud"):
        #     image_array = self.read_tif_from_gcs(self.gcs,self.file_list[index])
        #     image = Image.fromarray(image_array)
        if self.mode == "local":
            image = self.get_nfs_idr_image(self.file_list, self.index)
        if self.mode == "gcp":
            image = self.get_gs_idr_image(self.gcs_token, self.file_list, self.index)
        # if image.ndim != 2:
        #     return None
        #     return np.full([2048, 2048], np.nan)
        if image == None:
            return None
            return np.full([2048, 2048], np.nan)

        image = make_size(image, size=[2048, 2048])
        image_np = np.array(image, dtype=np.uint8)

        # print(f"Found image at {index}")
        if self.transform:
            trans = self.transform(image_np)
            return (trans, trans)
        return (image_np, image_np)

    def __len__(self):
        # return 100
        if self.mode == "http":
            return 100000
        else:
            return len(self.file_list)

    # def get_idr_image(self, imageid=171499):
    #     try:
    #         # conn = connection("idr.openmicroscopy.org", verbose=0)
    #         # print(f"Get image {str(imageid)}")
    #         # image_wrapped = conn.getObject("Image", imageid)
    #         # if image_wrapped == None:
    #             # return None
    #     except:
    #         return None
    #     # image_wrapped.getPrimaryPixels()
    #     try:
    #         image_plane = image_wrapped.getPrimaryPixels().getPlane(0)
    #         norm_image_plane = ((image_plane / image_plane.max()) * 255).astype(np.int8)
    #         pil_image = Image.fromarray(norm_image_plane, "L")
    #         # if image_plane == None:
    #         return pil_image
    #     except:
    #         return None
    #     file_path = file_list[index]
    #     return Image.open(file_path)

    def get_nfs_idr_image(self, file_list=[], index=0):
        file_path = file_list[index]
        return Image.open(file_path)

    def get_gs_idr_image(self, gcs_token, file_list=[], index=0):
        return self.get_gs_image(gcs_token, file_list[index])

    def get_gs_image(self, gcs, path):
        gcs = gcsfs.GCSFileSystem(
            token=tokengcs, access="read_only", skip_instance_cache=True
        )
        image_raw = gcs.cat(path)
        # current_image = Image.open(io.BytesIO(image_raw))
        return Image.open(io.BytesIO(image_raw))

    # def read_tif_from_gcs(self, gcs, relative_file_path):
    #     current_image = gcs.cat(relative_file_path)
    #     return imageio.core.asarray(imageio.imread(current_image, "TIFF"))

    # def __iter__(self):
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None:  # single-process data loading, return the full iterator
    #         iter_start = self.start
    #         iter_end = self.end
    #     else:  # in a worker process
    #         # split workload
    #         per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
    #         worker_id = worker_info.id
    #         iter_start = self.start + worker_id * per_worker
    #         iter_end = min(iter_start + per_worker, self.end)
    #     return iter(range(iter_start, iter_end))


# RETURN_NONE = True
# np.full([2048, 2048], np.nan)
# def get_idr_image(imageid=171499):
#     # conn = connection("idr.openmicroscopy.org")
#     image_wrapped = conn.getObject("Image", imageid)
#     if image_wrapped == None:
#         return None
#     # image_wrapped.getPrimaryPixels()
#     try:
#         image_plane = image_wrapped.getPrimaryPixels().getPlane(0)
#         norm_image_plane = ((image_plane/image_plane.max())*255).astype(np.int8)
#         pil_image = Image.fromarray(norm_image_plane,"L")
#         image = make_size(pil_image, size=[2048, 2048])
#         # if image_plane == None:
#         return image
#     except:
#         return None


def make_size(im, size=[2048, 2048]):
    if list(im.size) < list(size):
        image = pad_to(im, size)
    else:
        image = crop_to(im, size)
    return image


def pad_to(im, size=[2048, 2048]):
    left = int(size[0] / 2 - im.size[0] / 2)
    top = int(size[1] / 2 - im.size[1] / 2)

    image = Image.new(im.mode, size, 0)
    image.paste(im, (left, top))
    return image


def crop_to(im, size=[2048, 2048]):
    left = int(size[0] / 2 - im.size[0] / 2)
    upper = int(size[1] / 2 - im.size[1] / 2)
    right = left + size[0]
    lower = upper + size[1]
    image = im.crop((left, upper, right, lower))
    return image

    # Define a `worker_init_fn` that configures each dataset copy differently
    # def worker_init_fn(worker_id):
    #     worker_info = torch.utils.data.get_worker_info()
    #     dataset = worker_info.dataset  # the dataset copy in this worker process
    #     overall_start = dataset.start
    #     overall_end = dataset.end
    #     # configure the dataset to only process the split workload
    #     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    #     worker_id = worker_info.id
    #     dataset.start = overall_start + worker_id * per_worker
    #     dataset.end = min(dataset.start + per_worker, overall_end)


from torchvision import transforms

# transforms = ToTensor()  # we need this to convert PIL images to Tensor
# shuffle = True
transform = torch.nn.Sequential()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop((512, 512)),
        transforms.Normalize(0, 1),
    ]
)

dataset = IDRDataSet(
    transform=transform,
    training_dir=training_dir,
    glob_pattern=glob_pattern,
    mode=mode,
)

# import nonechucks as nc


# dataset = nc.SafeDataset(dataset)
# dataloader = nc.SafeDataLoader(dataset, batch_size=bs, shuffle=shuffle)
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# def collate_fn(batch):
#     len_batch = len(batch) # original batch length
#     batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
#     if len_batch > len(batch): # source all the required samples from the original dataset at random
#         diff = len_batch - len(batch)
#         for i in range(diff):
#             batch.append(dataset[np.random.randint(0, len(dataset))])

#     return torch.utils.data.dataloader.default_collate(batch)
dataloader = DataLoader(
    dataset,
    # batch_size=16,
    # shuffle=shuffle,
    collate_fn=collate_fn,
)

# dataloader = nc.SafeDataset(dataloader)
# data = iter(dataloader)
# a, b = data
# conn = connection(host="wss://idr.openmicroscopy.org/", user="public", password="public", port=443, verbose=0)

# conn = connection('idr.openmicroscopy.org')
import random

rgb, gt = random.choice(dataset)

plt.figure(figsize=(10, 5))
plt.subplot(221)
plt.imshow(torch.squeeze(rgb))
plt.title("RGB")
plt.subplot(222)
plt.imshow(torch.squeeze(gt))
plt.title("GT")

for i, (rgb, gt) in enumerate(dataloader):
    print(f"batch {i+1}:")
    # some plots
    # for i in range(bs):
    plt.figure(figsize=(10, 5))
    plt.subplot(221)
    plt.imshow(torch.squeeze(rgb[0, 0]))
    plt.title(f"RGB img{i+1}")
    plt.subplot(222)
    plt.imshow(torch.squeeze(gt[0, 0]))
    plt.title(f"GT img{i+1}")
    break
    # break
    # plt.show()
#  %%
dataloader = DataLoader(
    dataset,
    # batch_size=batch_size,
    # shuffle=shuffle,
    collate_fn=collate_fn,
    # num_workers=num_workers,
)
# %%
# import numpy as np
# import matplotlib.pyplot as plt
# def get_idr_image(imageid = 171499):
#     conn = connection('idr.openmicroscopy.org')
#     image_wrapped = conn.getObject("Image", imageid).getPrimaryPixels()
#     image_plane = image_wrapped.getPlane(0)
#     conn.close()
#     return image_plane

# def get_idr_image(imageid = 171499):
#     conn = connection('idr.openmicroscopy.org')
#     image_wrapped = conn.getObject("Image", imageid).getPrimaryPixels()
#     image_plane = image_wrapped.getPlane(0)
#     conn.close()
#     yield image_plane
# plt.imshow(dataset.get_idr_image(4684936))


#  %%
from torch.utils.data import Dataset, DataLoader


#  Unet structure outright stolen from the internet
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 1, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 1, out_channels, 3, 1)

        self.encoder = nn.Sequential(self.conv1, self.conv2, self.conv3)

        self.decoder = nn.Sequential(self.upconv3, self.upconv2, self.upconv1)

    # Call is essentially the same as running "forward"
    def __call__(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # downsampling part
        # conv1 = self.conv1(x)
        # conv2 = self.conv2(conv1)
        # conv3 = self.conv3(conv2)

        # upconv3 = self.upconv3(conv3)

        # upconv2 = self.upconv2(upconv3)
        # upconv1 = self.upconv1(upconv2)

        return x

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        return expand


class Net(nn.Module):
    def __init__(self, batch_size=16, n_class=1):
        super(Net, self).__init__()
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(1, 1, 1, 1)
        self.dp = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(1, 1, 1, 1)
        self.conv3 = nn.Conv2d(1, 1, 5, 5)
        self.fc2 = nn.AdaptiveAvgPool2d((100, 100))
        self.fc1 = nn.AdaptiveAvgPool2d((16, 16))
        self.conv4 = nn.Conv2d(1, 1, 1, 1)
        self.unet = UNet(1, 1)

    def forward(self, x):
        x = self.unet(x)

        output = x
        return output


#  Test Unet is working correctly by feeding in random data with the correct dimensionality
#  One channel in and out
#  You can also use channels to encode time which is useful
unet = UNet(1, 1)
x = torch.randn(16, 1, 128, 128)
unet(x).shape

#  Testing full model
model = Net(1, 1)
x = torch.randn(16, 1, 128, 128)
model(x).shape


# %% Define model and push to gpu (if availiable)
model = Net(1, 1).to(device)
#  Test
x = torch.randn(16, 1, 128, 128).to(device)

output = model(x)
output.shape


samples = 10000  # Number of samples per epoch, any number will do
# dataset = NPCDataSet(samples) # Generate callable datatset (it's an iterator)
batch_size = 32  # Publications claim that smaller is better for batch
# model = Net(batch_size).to(device)#
# %% Define model and push to gpu (if availiable)
from torch.utils.tensorboard import SummaryWriter

model = Net(batch_size, 1).to(device)

loader = dataloader
i = 0
# Make a run output folder for convenience
writer = SummaryWriter()
torch.manual_seed(42)

# %%
# Always set a manual seed else your runs will be incomparable

# batch_size = 16
log_interval = 10  # Logging interval
optimizer = torch.optim.Adam(
    model.parameters(),
)  # Adam does better than SGD
# optimizer = torch.optim.SGD(model.parameters(), lr=lr) # Adam does better than SGD

model.train()  # Set model to training mode (probably deprecated?)
# loss_fn = nn.MSELoss(reduction='mean')
loss_fn = nn.MSELoss()  # MSE is fine for this

# loss_fn = nn.BCELoss()
# loss_fn = nn.BCEWithLogitsLoss()


# for epoch in range(0, epochs):
#     for batch_idx, (inputs, outputs) in enumerate(loader):

#         data = inputs[0].view(1,1,512,512)
#         target = outputs[0].view(1,1,512,512)

#         optimizer.zero_grad()

#         for dwell in range(0, 100):
#             output = model(data)
#             loss = loss_fn(output, target)
#             # loss = torch.nn.functional.nll_loss( output, target)
#             loss.backward()
#             optimizer.step()
#             i += 1
#             writer.add_scalar("Loss/train", loss, i)
#             writer.add_image("output", data.view(1,512,512))
#             writer.add_image("target", target.view(1,512,512))
#         if batch_idx % log_interval == 0:
#             print(
#                 f"Train Epoch: {str(epoch)} {str(batch_idx)} \t Loss: {str(loss.item())}"
#             )
# %%
# torch.save(model.state_dict(), "model")
# #  %%
# plt.imshow(output.cpu().detach()[0, 0, :, :])
# #  %%
# plt.imshow(target.cpu().detach()[0, 0, :, :])

# #  %%

# plt.imshow(inputs.cpu().detach()[0, :, :])


#  %%
# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# from torch.utils.data import random_split
# from torchvision.datasets import MNIST
# from torchvision import transforms


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, batch_size=1, learning_rate=1e-3):
        super().__init__()
        self.encoder = Net(batch_size, 1)
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, outputs = train_batch
        data = inputs[0].view(1, 1, 512, 512)
        target = outputs[0].view(1, 1, 512, 512)
        # optimizer.zero_grad()
        output = self.encoder(data)
        # x, y = train_batch
        # x = x.view(x.size(0), -1)
        loss = loss_fn(output, target)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        tensorboard = self.logger.experiment
        tensorboard.add_scalar("Loss/train", loss, batch_idx)
        tensorboard.add_image("output", output.view(1, 512, 512), batch_idx)
        tensorboard.add_image("target", target.view(1, 512, 512), batch_idx)

        return loss

    # def backward(self, use_amp, loss, optimizer):
    #     # if use_amp:
    #     #     with amp.scale_loss(loss, optimizer) as scaled_loss:
    #     #         scaled_loss.backward()
    #     # else:
    #     # optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    # def validation_step(self, val_batch, batch_idx):
    # 	x, y = val_batch
    # 	x = x.view(x.size(0), -1)
    # 	z = self.encoder(x)
    # 	x_hat = self.decoder(z)
    # 	loss = F.mse_loss(x_hat, x)
    # 	self.log('val_loss', loss)


# data


model = LitAutoEncoder()


tb_logger = pl_loggers.TensorBoardLogger("runs/")
# training
# tb_logger = pl_loggers.TensorBoardLogger('logs/')

checkpoint_callback = ModelCheckpoint(every_n_train_steps=100)

trainer = pl.Trainer(
    logger=tb_logger,
    gpus=gpus,
    accumulate_grad_batches=1,
    callbacks=[checkpoint_callback],
).from_argparse_args(args)
#  %%

trainer.fit(model, dataloader)

#  %%
#
# # %%
# import torch
# import torchvision
# from torch import nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image
# from torchvision.datasets import MNIST
# import os

# # https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py

# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#             nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
#             nn.Tanh()
#         )

#     def forward(self, x):
# # %%
# import torch
# import torchvision
# from torch import nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image
# from torchvision.datasets import MNIST
# import os

# # https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py

# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#             nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

# num_epochs = 100
# batch_size = 128
# learning_rate = 1e-3


# model = autoencoder().cuda()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
#                              weight_decay=1e-5)
# total_loss = 0
# for epoch in range(num_epochs):
#     for data in dataloader:
#         img, _ = data
#         img = Variable(img).cuda()
#         # ===================forward=====================
#         output = model(img)
#         loss = criterion(output, img)
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================log========================
#     total_loss += loss.data
#     print(f'epoch [{epoch+1}/{num_epochs}], loss:{total_loss}')
#     if epoch % 10 == 0:
#         pic = to_img(output.cpu().data)
#         save_image(pic, f'./dc_img/image_{str(epoch)}.png')

# torch.save(model.state_dict(), './conv_autoencoder.pth')
#         x = self.decoder(x)
#         return x

# num_epochs = 100
# batch_size = 128
# learning_rate = 1e-3


# model = autoencoder().cuda()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
#                              weight_decay=1e-5)
# total_loss = 0
# for epoch in range(num_epochs):
#     for data in dataloader:
#         img, _ = data
#         img = Variable(img).cuda()
#         # ===================forward=====================
#         output = model(img)
#         loss = criterion(output, img)
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================log========================
#     total_loss += loss.data
#     print(f'epoch [{epoch+1}/{num_epochs}], loss:{total_loss}')
#     if epoch % 10 == 0:
#         pic = to_img(output.cpu().data)
#         save_image(pic, f'./dc_img/image_{str(epoch)}.png')

# torch.save(model.state_dict(), './conv_autoencoder.pth')
# %%
