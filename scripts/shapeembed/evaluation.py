from torchvision import datasets, transforms
import pyefd
from umap import UMAP
from skimage import measure
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import StandardScaler
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate, cross_val_predict, KFold, train_test_split, StratifiedKFold

import tqdm
import numpy
import pandas
import logging
import seaborn
import matplotlib.pyplot as plt

from bioimage_embed.shapes.transforms import ImageToCoords

# logging facilities
###############################################################################
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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
  df['class'] = all_lbls
  df.dropna()
  return df

def run_kmeans(dataframe, random_seed=42):
  # run KMeans and derive accuracy metric and confusion matrix
  kmeans = KMeans( n_clusters=len(dataframe['class'].unique())
                 , random_state=random_seed
                 ).fit(dataframe.drop('class', axis=1))
  accuracy = accuracy_score(dataframe['class'], kmeans.labels_)
  conf_mat = confusion_matrix(dataframe['class'], kmeans.labels_)
  return kmeans, accuracy, conf_mat

def run_regionprops( dataset_params
                   , properties = [ "area"
                                  , "perimeter"
                                  , "centroid"
                                  , "major_axis_length"
                                  , "minor_axis_length"
                                  , "orientation" ] ):
  # access the dataset
  assert dataset_params.type == 'mask', f'unsupported dataset type {dataset_params.type}'
  ds = datasets.ImageFolder(dataset_params.path, transforms.Grayscale(1))
  # ... and run regionprops for the given properties for each image
  dfs = []
  logger.info(f'running regionprops on {dataset_params.name}')
  logger.info(f'({dataset_params.path})')
  for i, (img, lbl) in enumerate(tqdm.tqdm(ds)):
    data = numpy.where(numpy.array(img)>20, 255, 0)
    t = measure.regionprops_table(data, properties=properties)
    df = pandas.DataFrame(t)
    assert df.shape[0] == 1, f'More than one object in image #{i}'
    df.index = [i]
    df['class'] = lbl
    #df.set_index("class", inplace=True)
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
  df = pandas.concat(dfs).xs('coeffs', level='coeffs')
  df.reset_index(level='class', inplace=True)
  return df

def score_dataframe( df, name
                   , test_sz=0.2, rand_seed=42, shuffle=True, k_folds=5 ):
  # drop strings and python object columns
  #clean_df = df.select_dtypes(exclude=['object'])
  clean_df = df.select_dtypes(include=['number'])
  # TODO, currently unused
  # Split the data into training and test sets
  #X_train, X_test, y_train, y_test = train_test_split(
  #  clean_df.drop('class', axis=1), clean_df['class']
  #, stratify=clean_df['class']
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
  # build confusion matrix
  clean_df.columns = clean_df.columns.astype(str) # only string column names
  lbl_pred = cross_val_predict( pipeline
                              , clean_df.drop('class', axis=1)
                              , clean_df['class'])
  conf_mat = confusion_matrix(clean_df['class'], lbl_pred)
  # Perform k-fold cross-validation
  cv_results = cross_validate(
    estimator=pipeline
  , X=clean_df.drop('class', axis=1)
  , y=clean_df['class']
  , cv=StratifiedKFold(n_splits=k_folds)
  , scoring=scoring
  , n_jobs=-1
  , return_train_score=False
  )
  # Put the results into a DataFrame
  df = pandas.DataFrame(cv_results)
  df = df.drop(["fit_time", "score_time"], axis=1)
  df.insert(loc=0, column='trial', value=name)
  return conf_mat, df

def umap_plot( df
             , name
             , outputdir='.'
             , n_neighbors=15
             , min_dist=0.1
             , n_components=2
             , rand_seed=42
             , split=0.7
             , width=3.45
             , height=3.45 / 1.618 ):
  clean_df = df.select_dtypes(include=['number'])
  umap_reducer = UMAP( n_neighbors=n_neighbors
                     , min_dist=min_dist
                     , n_components=n_components
                     , random_state=rand_seed )
  mask = numpy.random.rand(len(clean_df)) < split

  #clean_df.reset_index(level='class', inplace=True)
  classes = clean_df['class'].copy()
  semi_labels = classes.copy()
  semi_labels[~mask] = -1  # Assuming -1 indicates unknown label for semi-supervision
  clean_df.drop('class', axis=1, inplace=True)

  umap_embedding = umap_reducer.fit_transform(clean_df, y=semi_labels)
  umap_data=pandas.DataFrame(umap_embedding, columns=["umap0", "umap1"])
  umap_data['class'] = classes

  ax = seaborn.relplot( data=umap_data
                      , x="umap0"
                      , y="umap1"
                      , hue="class"
                      , palette="deep"
                      , alpha=0.5
                      , edgecolor=None
                      , s=5
                      , height=height
                      , aspect=0.5 * width / height )

  seaborn.move_legend(ax, "upper center")
  ax.set(xlabel=None, ylabel=None)
  seaborn.despine(left=True, bottom=True)
  plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
  plt.tight_layout()
  plt.savefig(f"{outputdir}/umap_{name}.pdf")
  plt.close()

def save_scores( scores_df
               , outputdir='.'
               , width = 3.45
               , height = 3.45 / 1.618 ):
  # save all raw scores as csv
  scores_df.to_csv(f"{outputdir}/scores_df.csv")
  # save score means as csv
  scores_df.groupby("trial").mean().to_csv(f"{outputdir}/scores_df_mean.csv")
  # save a barplot representation of scores
  melted_df = scores_df.melt( id_vars="trial"
                            , var_name="Metric"
                            , value_name="Score" )
  seaborn.catplot( data=melted_df
                 , kind="bar"
                 , x="trial"
                 , hue="Metric"
                 , y="Score"
                 , errorbar="se"
                 , height=height
                 , aspect=width * 2**0.5 / height )
  plt.savefig(f"{outputdir}/scores_barplot.pdf")
  plt.close()
  # log info
  logger.info(melted_df.set_index(["trial", "Metric"])
                .xs("test_f1", level="Metric", drop_level=False)
                .groupby("trial")
                .mean())
