from torchvision import datasets, transforms
import pyefd
from skimage import measure
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import StandardScaler
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate, KFold, train_test_split, StratifiedKFold

import tqdm
import numpy
import pandas
import logging

from bioimage_embed.shapes.transforms import ImageToCoords

# logging facilities
###############################################################################
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def dataloader_to_dataframe(dataloader):
  # gather the data and the associated labels, and drop rows with NaNs
  all_data = []
  all_lbls = []
  for batch in dataloader:
    inputs, lbls = batch
    for data, lbl in zip(inputs, lbls):
      all_data.append(data.flatten().numpy())
      all_lbls.append(int(lbl))
  df = pandas.DataFrame(all_data)
  df['label'] = all_lbls
  df.dropna()
  return df

def run_kmeans(dataframe, random_seed=42):
  # run KMeans and derive accuracy metric and confusion matrix
  kmeans = KMeans( n_clusters=len(dataframe['label'].unique())
                 , random_state=random_seed
                 ).fit(dataframe.drop('label', axis=1))
  accuracy = accuracy_score(dataframe['label'], kmeans.labels_)
  conf_mat = confusion_matrix(dataframe['label'], kmeans.labels_)

  return kmeans, accuracy, conf_mat

def run_regionprops( dataset_params
                   , properties = [ "area"
                                  , "perimeter"
                                  , "centroid"
                                  , "major_axis_length"
                                  , "minor_axis_length"
                                  , "orientation" ] ):
  # access the dataset
  assert dataset_params.type == 'mask'
  ds = datasets.ImageFolder(dataset_params.path, transforms.Grayscale(1))
  # ... and run regionprops for the given properties for each image
  dfs = []
  logger.info(f'running regionprops on {dataset_params.name}')
  logger.info(f'({dataset_params.path})')
  for i, (img, lbl) in enumerate(tqdm.tqdm(ds)):
    t = measure.regionprops_table(numpy.array(img), properties=properties)
    df = pandas.DataFrame(t)
    df['class'] = lbl
    df.set_index("class", inplace=True)
    dfs.append(df)
  # concatenate results as a single dataframe and return it
  df = pandas.concat(dfs)
  return df

def run_elliptic_fourier_descriptors(dataset_params, contour_size=512):
  # access the dataset
  assert dataset_params.type == 'mask'
  ds = datasets.ImageFolder( dataset_params.path
                           , transform=transforms.Compose([
                               transforms.Grayscale(1)
                             , ImageToCoords(contour_size) ]))
  # ... and run efd on each image
  dfs = []
  logger.info(f'running efd on {dataset_params.name}')
  logger.info(f'({dataset_params.path})')
  for i, (img, lbl) in enumerate(tqdm.tqdm(ds)):
    coeffs = pyefd.elliptic_fourier_descriptors(img, order=10, normalize=False)
    norm_coeffs = pyefd.normalize_efd(coeffs)
    df = pandas.DataFrame({
      "norm_coeffs": norm_coeffs.flatten().tolist()
    , "coeffs": coeffs.flatten().tolist()
    }).T.rename_axis("coeffs")
    df['class'] = lbl
    df.set_index("class", inplace=True, append=True)
    dfs.append(df)
  # concatenate results as a single dataframe and return it
  return pandas.concat(dfs).xs('coeffs', level='coeffs')

def score_dataframe(df, test_sz=0.2, rand_seed=42, shuffle=True, k_folds=5):
  # TODO, currently unused
  # Split the data into training and test sets
  #X_train, X_test, y_train, y_test = train_test_split(
  #  df, df.index, stratify=df.index
  #, test_size=test_sz, randm_state=rand_seed, shuffle=shuffle
  #)
  # Define a dictionary of metrics
  scoring = {
    "accuracy": make_scorer(metrics.balanced_accuracy_score)
  , "precision": make_scorer(metrics.precision_score, average="macro")
  , "recall": make_scorer(metrics.recall_score, average="macro")
  , "f1": make_scorer(metrics.f1_score, average="macro")
  #, "roc_auc": make_scorer(metrics.roc_auc_score, average="macro")
  }
  # Create a random forest classifier
  pipeline = Pipeline([
    ("scaler", StandardScaler())
  #, ("pca", PCA(n_components=0.95, whiten=True, random_state=rand_seed))
  , ("clf", RandomForestClassifier())
  #, ("clf", DummyClassifier())
  ])
  # Perform k-fold cross-validation
  cv_results = cross_validate(
    estimator=pipeline
  , X=df
  , y=df.index
  , cv=StratifiedKFold(n_splits=k_folds)
  , scoring=scoring
  , n_jobs=-1
  , return_train_score=False
  )
  # Put the results into a DataFrame
  return pandas.DataFrame(cv_results)
