import torch
import umap
import pandas as pd
from torchvision import transforms
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import umap.plot
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# import os
import glob
from idr import connection

model_path = "models/model.pt"

# Load the TorchScript model
model = torch.jit.load(model_path)

# Use the loaded model for inference or other tasks
# output = model(torch.randn(1, 1, 512, 512))
encoder = model.model._encoder
# output = encoder(torch.randn(1, 1, 512, 512))


transform = transforms.Compose(
    [
        transforms.Grayscale(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine((0, 360)),
        transforms.RandomResizedCrop(size=512),
        # transforms.RandomCrop(size=(512,512)),
        # transforms.GaussianBlur(5),
        transforms.ToTensor(),
        # transforms.Normalize((0.485), (0.229)),
    ]
)


# Create a connection
print("Trying to connect to IDR...")
conn = connection("idr.openmicroscopy.org", "public", "public")

# Define the folder
folder = "/home/ctr26/gdrive/+projects/ai_data/data/idr/inference"

# Get the list of .tiff files in the folder and subfolders
glob_strs = ["*", "0_3_0", "0_2_0", "0_1_0", "0_0_0"]
image_name = glob_strs[-1]
# for glob_str in glob_strs:
glob_str = f"/**/{image_name}.tiff"
# seed = 42
filenames = glob.glob(folder + glob_str, recursive=True)
# filenames = np.random.RandomState(seed).permutation(filenames)
# Initialize lists for images and labels
images = []
labels = []
from io import BytesIO

# Loop over the filenames
# for filename in filenames[0:500]:
#     # Open the image and convert to PyTorch tensor
#     with open(filename, "rb") as f:
#         image_data = f.read()
#     image = Image.open(BytesIO(image_data))
#     # image = Image.open(filename)
#     # image = ToTensor()(image)
#     images.append(image)

#     # Extract the image ID from the filename
#     image_id = os.path.basename(os.path.dirname(filename))

#     # Get the image data
#     image_data = conn.getObject("Image", image_id)
#     well_id = image_data.getParent().getWell().getId()
#     well = conn.getObject("Well", well_id)
#     metadata = well.getAnnotation().getValue()
#     try:
#         gene_symbol = dict(metadata)["Gene Symbol"]
#     except:
#         gene_symbol = "Wildtype"

#     # well.getAnnotation().getValue()
#     # gene_id = image_data.getPrimaryAnnotatedTerms()[0]['id']
#     labels.append(gene_symbol)


# Now you have your images and labels, you can create a PyTorch dataset:
class GeneData:
    # def __init__(self, images, labels):
    #     self.images = images
    #     self.labels = labels
    def __init__(self, transform):
        self.transform = transform

    # def __len__(self):
    # return len(self.images)

    # def __getitem__(self, filename):
    def __call__(self, filename):
        # filename = filename
        z = self.transform(self.get_image(filename))
        label = self.get_label(filename)
        return z, label
        # return self.images[idx], self.labels[idx]

    def get_image(self, filename):
        with open(filename, "rb") as f:
            image_data = f.read()
        image = Image.open(BytesIO(image_data))
        # image = Image.open(filename)
        # image = ToTensor()(image)
        return image
        # images.append(image)

    def get_embedding(self, filename):
        image = self.get_image(filename)
        z = self.encode(image)
        return z

    def encode(self, image):
        tensor = transform(image)
        z = encoder(tensor.unsqueeze(0).to(torch.float))
        return z
        # Extract the image ID from the filename

    def get_image_id(self, filename):
        image_id = os.path.basename(os.path.dirname(filename))
        return image_id

    def image_id_to_label(self, image_id):
        image_data = conn.getObject("Image", image_id)
        well_id = image_data.getParent().getWell().getId()
        well = conn.getObject("Well", well_id)
        metadata = well.getAnnotation().getValue()
        try:
            gene_symbol = dict(metadata)["Gene Symbol"]
        except:
            gene_symbol = "Wildtype"
        return gene_symbol

    def get_label(self, filename):
        image_id = self.get_image_id(filename)
        label = self.image_id_to_label(image_id)
        return label


# dataset = GeneDataset(images, labels)
# for filename in filenames[0:500]:
# dataset
# GeneData(transform)(filenames[0])
# dataset = [GeneData(transform)(filename) for filename in filenames[0:500]]
df = pd.DataFrame(index=filenames[0:10])
import dask.dataframe as dd


def genedata_to_series(filename, transform):
    z, label = GeneData(transform)(filename)
    return pd.Series({"z": z.numpy().flatten(), "label": label})
    # return pd.DataFrame({"z":z.numpy().flatten(),
    #   "label":label})

    return z.numpy().flatten(), label


def genedata_row_to_series(row, transform):
    return genedata_to_series(row["filenames"], transform)


df = pd.DataFrame(data={"filenames": filenames[0:10]})
df[["z", "label"]] = df.apply(genedata_row_to_series, axis=1, transform=transform)


ddf = dd.from_pandas(pd.DataFrame(data={"filenames": filenames}), npartitions=32)
result = ddf.apply(
    genedata_row_to_series,
    transform=transform,
    axis=1,
    meta=pd.DataFrame({"z": [], "label": str}),
)
from dask.diagnostics import ProgressBar

# Apply the function to each row of the Dask DataFrame
if not os.path.isfile("z.csv"):
    with ProgressBar():
        ddf[["z", "label"]] = result
        df = ddf.compute()
        df = df.set_index(["filenames", "label"]).apply(pd.Series)
        df = df["z"].apply(pd.Series)
        # df.to_csv("z.csv",index=False)

# z = pd.read_csv("z.csv").set_index(["filenames", "label"])
z = df

top_10_labels = df.index.get_level_values("label").value_counts().head(10).index
z = df[df.index.get_level_values("label").isin(top_10_labels)]


# X = torch.stack(z).detach().numpy().reshape(500, -1)
X = z.to_numpy()
labels = z.index.get_level_values("label").to_numpy().astype(str)

# Create a UMAP object and fit-transform the data
reducer = umap.UMAP()

# Convert the target variable to numeric labels using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# projection = reducer.fit_transform(X, y=y)
mapper = reducer.fit(X, y=y)

umap.plot.points(mapper, labels=labels)
# umap.plot.savefig(f"{image_name}.png")
plt.savefig(f"umap_{image_name}.png")
plt.show()

conn.close()

from sklearn.decomposition import PCA

# Create a pipeline with PCA and Random Forest classifier
pipeline = Pipeline(
    [
        (
            "pca",
            PCA(n_components=0.95),
        ),  # Set the number of desired components for PCA
        ("classifier", RandomForestClassifier()),
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


# Predict labels for the validation set
y_pred = classifier.predict(X_test)

# Evaluate the classifier using classification report
classification_metrics = classification_report(y_test, y_pred)
print(classification_metrics)
