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

# logging facilities
###############################################################################
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

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

def score_dataframe( df, name
                   , tag_columns=[]
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
  tag_columns.reverse()
  for tag_col_name, tag_col_value in tag_columns:
    df.insert(loc=0, column=tag_col_name, value=tag_col_value)
  return conf_mat, df

def confusion_matrix_plot( cm, name, outputdir
                         , figsize=(10,7) ):
  # Plot confusion matrix
  plt.clf()  # Clear figure
  plt.figure(figsize=figsize)
  seaborn.heatmap(cm, annot=True, fmt='d')
  plt.title(f'{name} - Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.savefig(f'{outputdir}/{name}-confusion_matrix.png')
  plt.clf()  # Clear figure

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
  mask = numpy.random.rand(clean_df.shape[0]) < split

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
  plt.savefig(f"{outputdir}/{name}-umap.pdf")
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

def save_barplot( scores_df
                , outputdir='.'
                , width = 7
                , height = 7 / 1.2 ):
  # save a barplot representation of scores
  melted_df = scores_df[['model', 'beta', 'compression_factor', 'latent_dim', 'batch_size', 'test_f1']].melt(
    id_vars=['model', 'beta', 'compression_factor', 'latent_dim', 'batch_size']
  , var_name="Metric"
  , value_name="Score"
  )
  # test plots...
  for m in melted_df['model'].unique():
    # 1 - general overview plot...
    df = melted_df.loc[ (melted_df['model'] == m)
                      , ['compression_factor', 'latent_dim', 'batch_size', 'beta', 'Metric', 'Score'] ].sort_values(by=['compression_factor', 'latent_dim', 'batch_size', 'beta'])
    hue = df[['compression_factor', 'latent_dim']].apply(lambda r: f'cf: {r.compression_factor}({r.latent_dim})', axis=1)
    if 'beta' in m:
      hue = df[['compression_factor', 'latent_dim', 'beta']].apply(lambda r: f'cf: {r.compression_factor}({r.latent_dim}), beta: {r.beta}', axis=1)
    ax = seaborn.catplot( data=df
                        , kind="bar"
                        , x='batch_size'
                        , y="Score"
                        , hue=hue
                        , errorbar="se"
                        , height=height
                        , aspect=width * 2**0.5 / height )
    #ax.tick_params(axis='x', rotation=90)
    #ax.set(xlabel=None)
    #ax.set(xticklabels=[])
    ax._legend.remove()
    #ax.fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3)
    #ax.fig.legend(ncol=4, loc='lower center')
    ax.fig.legend(ncol=1)
    #ax.fig.subplots_adjust(top=0.9)
    #ax.set(title=f'f1 score against batch size ({m})')

    #add overall title
    plt.title(f'f1 score against batch size ({m})', fontsize=16)

    ##add axis titles
    #plt.xlabel('')
    #plt.ylabel('')

    #rotate x-axis labels
    #plt.xticks(rotation=45)

    plt.savefig(f"{outputdir}/barplot_{m}_x_bs.pdf", bbox_inches="tight")
    plt.close()

    # 1b - general overview plot...
    df = melted_df.loc[ (melted_df['model'] == m)
                      , ['batch_size', 'compression_factor', 'latent_dim', 'beta', 'Metric', 'Score'] ].sort_values(by=['batch_size', 'compression_factor', 'latent_dim', 'beta'])
    hue = df['batch_size'].apply(lambda r: f'bs: {r}')
    if 'beta' in m:
      hue = df[['batch_size', 'beta']].apply(lambda r: f'bs: {r.batch_size}, beta: {r.beta}', axis=1)
    ax = seaborn.catplot( data=df
                        , kind="bar"
                        , x=df[['compression_factor', 'latent_dim']].apply(lambda r: f'cf: {r.compression_factor}({r.latent_dim})', axis=1)
                        , y="Score"
                        , hue=hue
                        , errorbar="se"
                        , height=height
                        , aspect=width * 2**0.5 / height )
    #ax.tick_params(axis='x', rotation=90)
    #ax.set(xlabel=None)
    #ax.set(xticklabels=[])
    ax._legend.remove()
    #ax.fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3)
    #ax.fig.legend(ncol=4, loc='lower center')
    ax.fig.legend(ncol=1)
    #ax.fig.subplots_adjust(top=0.9)
    #ax.set(title=f'f1 score against batch size ({m})')

    #add overall title
    plt.title(f'f1 score against compression factor (latent space size) ({m})', fontsize=16)

    ##add axis titles
    #plt.xlabel('')
    #plt.ylabel('')

    #rotate x-axis labels
    #plt.xticks(rotation=45)

    plt.savefig(f"{outputdir}/barplot_{m}_x_cf.pdf", bbox_inches="tight")
    plt.close()

    # 2 - more specific plots
    for cf in melted_df['compression_factor'].unique():
      if 'beta' in m:
        for bs in melted_df['batch_size'].unique():
          ax = seaborn.catplot( data=melted_df.loc[ (melted_df['model'] == m) & (melted_df['compression_factor'] == cf) & (melted_df['batch_size'] == bs)
                                                  , ['beta', 'Metric', 'Score'] ]
                              , kind="bar"
                              , x='beta'
                              , hue="Metric"
                              , y="Score"
                              , errorbar="se"
                              , height=height
                              , aspect=width * 2**0.5 / height )
          ax.tick_params(axis='x', rotation=90)
          ax.fig.subplots_adjust(top=0.9)
          ax.set(title=f'f1 score against beta ({m}, compression factor {cf}, batch size {bs})')
          plt.savefig(f"{outputdir}/beta_barplot_{m}_{cf}_{bs}.pdf")
          plt.close()
      ax = seaborn.catplot( data=melted_df.loc[ (melted_df['model'] == m) & (melted_df['compression_factor'] == cf)
                                              , ['batch_size', 'beta', 'Metric', 'Score'] ]
                          , kind="bar"
                          , x='batch_size'
                          , hue='beta' if 'beta' in m else 'Metric'
                          , y="Score"
                          , errorbar="se"
                          , height=height
                          , aspect=width * 2**0.5 / height )
      ax.tick_params(axis='x', rotation=90)
      ax.fig.subplots_adjust(top=0.9)
      ax.set(title=f'f1 score against batch size ({m}, compression factor {cf})')
      plt.savefig(f"{outputdir}/barplot_{m}_x_bs_cf{cf}.pdf")
      plt.close()
      ax = seaborn.catplot( data=melted_df.loc[ (melted_df['model'] == m) & (melted_df['batch_size'] == cf)
                                              , ['compression_factor', 'beta', 'Metric', 'Score'] ]
                          , kind="bar"
                          , x='compression_factor'
                          , hue='beta' if 'beta' in m else 'Metric'
                          , y="Score"
                          , errorbar="se"
                          , height=height
                          , aspect=width * 2**0.5 / height )
      ax.tick_params(axis='x', rotation=90)
      ax.fig.subplots_adjust(top=0.9)
      ax.set(title=f'f1 score against batch size ({m}, compression factor {cf})')
      plt.savefig(f"{outputdir}/barplot_{m}_x_cf_bs{bs}.pdf")
      plt.close()
  # log info
  #logger.info(melted_df.set_index(["trial", "Metric"])
  #              .xs("test_f1", level="Metric", drop_level=False)
  #              .groupby("trial")
  #              .mean())
