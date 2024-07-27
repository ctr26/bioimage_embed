#! /usr/bin/env python3

import os
import re
import shutil
import logging
import seaborn
import argparse
import datetime
import functools
import pandas as pd

from common_helpers import *
from evaluation import *

#def simple_table(df, tname, model_re=".*vq.*"):
def simple_table(df, tname, model_re=".*", sort_by_col=None, ascending=False, best_n=40):
  cols=['model', 'compression_factor', 'latent_dim', 'batch_size', 'beta', 'test_f1', 'mse/test']
  df = df.loc[df.model.str.contains(model_re), cols].sort_values(by=cols)
  if sort_by_col:
    df = df.sort_values(by=sort_by_col, ascending=ascending)
  df = df.iloc[:best_n]

  with open(f'{tname}_tabular.tex', 'w') as fp:
    fp.write("\\begin{tabular}{|llll|r|r|} \hline\n")
    fp.write("Model & CF (and latent space size) & batch size & BETA & F1 score & Mse \\\\ \hline\n")
    for _, r in df.iterrows():
      mname = r['model'].replace('_','\_')
      beta = '-' if pd.isna(r['beta']) else r['beta']
      fp.write(f"{mname} & {r['compression_factor']} ({r['latent_dim']}) & {r['batch_size']} & {beta} & {r['test_f1']:f} & {r['mse/test']:f} \\\\\n")
    fp.write("\hline\n")
    fp.write("\end{tabular}\n")

def compare_f1_mse_table(df, tname, best_n=40):
  cols=['model', 'compression_factor', 'latent_dim', 'batch_size', 'beta', 'test_f1', 'mse/test']
  df0 = df[cols].sort_values(by=cols)
  df0 = df0.sort_values(by='test_f1', ascending=False)
  df0 = df0.iloc[:best_n]
  df1 = df[cols].sort_values(by=cols)
  df1 = df1.sort_values(by='mse/test', ascending=True)
  df1 = df1.iloc[:best_n]
  df = pd.concat([df0.reset_index(), df1.reset_index()], axis=1, keys=['f1', 'mse'])
  print(df)
  with open(f'{tname}_tabular.tex', 'w') as fp:
    fp.write("\\begin{tabular}{|llll|r|r|llll|r|r|} \hline\n")
    fp.write("\multicolumn{6}{|l}{Best F1 score} & \multicolumn{6}{|l|}{Best Mse} \\\\\n")
    fp.write("Model & CF (latent space) & batch size & BETA & F1 score & Mse & Model & CF (latent space) & batch size & BETA & F1 score & Mse \\\\ \hline\n")
    for _, r in df.iterrows():
      f1_name = r[('f1', 'model')].replace('_','\_')
      mse_name = r[('mse', 'model')].replace('_','\_')
      f1_beta = '-' if pd.isna(r[('f1', 'beta')]) else r[('f1', 'beta')]
      mse_beta = '-' if pd.isna(r[('mse', 'beta')]) else r[('mse', 'beta')]
      fp.write(f"{f1_name} & {r[('f1', 'compression_factor')]} ({r[('f1', 'latent_dim')]}) & {r[('f1', 'batch_size')]} & {f1_beta} & {r[('f1', 'test_f1')]:f} & {r[('f1', 'mse/test')]:f} & {mse_name} & {r[('mse', 'compression_factor')]} ({r[('mse', 'latent_dim')]}) & {r[('mse', 'batch_size')]} & {mse_beta} & {r[('mse', 'test_f1')]:f} & {r[('mse', 'mse/test')]:f} \\\\\n")
    fp.write("\hline\n")
    fp.write("\end{tabular}\n")

def main_process(clargs, logger=logging.getLogger(__name__)):

  params = []
  for f in clargs.run_folders:
    ps = find_existing_run_scores(f)
    for p in ps: p.folder = f
    params.append(ps)
  params = [x for ps in params for x in ps]
  logger.debug(params)

  os.makedirs(clargs.output_dir, exist_ok=True)

  dfs = []
  for p in params:

    # open scores dataframe
    df = pd.read_csv(p.csv_file, index_col=0)

    # split model column in case model args are present
    model_cols = df['model'].str.split('-', n=1, expand=True)
    if model_cols.shape[1] == 2:
      df = df.drop('model', axis=1)
      df.insert(1, 'model_args', model_cols[1])
      df.insert(1, 'model', model_cols[0])

    # pair up with confusion matrix
    conf_mat_file = f'{job_str(p)}-shapeembed-confusion_matrix.png'
    print(f'{p.folder}/{conf_mat_file}')
    if os.path.isfile(f'{p.folder}/{conf_mat_file}'):
      shutil.copy(f'{p.folder}/{conf_mat_file}',f'{clargs.output_dir}/{conf_mat_file}')
      df['conf_mat'] = f'./{conf_mat_file}'
    else:
      df['conf_mat'] = f'nofile'

    # pair up with umap
    umap_file = f'{job_str(p)}-shapeembed-umap.pdf'
    if os.path.isfile(f'{p.folder}/{umap_file}'):
      shutil.copy(f'{p.folder}/{umap_file}',f'{clargs.output_dir}/{umap_file}')
      df['umap'] = f'./{umap_file}'
    else:
      df['umap'] = f'nofile'

    ## pair up with barplot
    #barplot = f'scores_barplot.pdf'
    #if os.path.isfile(f'{d}/{barplot}'):
    #  shutil.copy(f'{d}/{barplot}',f'{clargs.output_dir}/{run_name}_{barplot}')
    #  df.loc[df['trial'] == trial, 'barplot'] = f'./{run_name}_{barplot}'
    #else:
    #  df.loc[df['trial'] == trial, 'barplot'] = f'nofile'

    # add dataframe to list for future concatenation
    dfs.append(df.convert_dtypes())

  # gather all dataframes together
  df = pd.concat(dfs)
  logger.debug(df)
  df.to_csv(f'{clargs.output_dir}/all_scores_df.csv', index=False)
  save_barplot(df, clargs.output_dir)

  #df = df.iloc[:, 1:] # drop first column 'unnamed' for non-mean df
  # define a Custom aggregation
  # function for finding total
  def keep_first_fname(series): 
    return functools.reduce(lambda x, y: y if x == 'nofile' else x, series)
  idx_cols = ['trial', 'dataset', 'model', 'compression_factor', 'latent_dim', 'batch_size']
  df.set_index(idx_cols, inplace=True)
  df.sort_index(inplace=True)
  #df = df.groupby(level=['trial', 'dataset', 'model', 'compression_factor', 'latent_dim', 'batch_size']).agg({
  df = df.groupby(level=idx_cols).agg({
    'beta': 'mean'
  , 'test_accuracy': 'mean'
  , 'test_precision': 'mean'
  , 'test_recall': 'mean'
  , 'test_f1': 'mean'
  , 'mse/test': 'mean'
  , 'loss/test': 'mean'
  , 'mse/val': 'mean'
  , 'loss/val': 'mean'
  , 'conf_mat': keep_first_fname
  , 'umap': keep_first_fname
  #, 'barplot': keep_first_fname
  })

  print('-'*80)
  print(df)
  print('-'*80)
  df.to_csv(f'{clargs.output_dir}/all_scores_agg_df.csv')
  df = df.reset_index()

  # table results for f1 and mse comparison
  simple_table(df, f'{clargs.output_dir}/table_top40_f1', sort_by_col='test_f1')
  simple_table(df, f'{clargs.output_dir}/table_top40_mse', sort_by_col='mse/test', ascending=True)
  compare_f1_mse_table(df, f'{clargs.output_dir}/table_top5_compare', best_n=5)

  # mse / f1 plots
  dff=df[df['mse/test']<df['mse/test'].quantile(0.9)] # drop mse outlier
  #mse=df['mse/test']
  #print(f'mse, mean: {mse.mean()}, std: {mse.std()}')
  ax = seaborn.relplot(data=dff, x='mse/test', y='test_f1', hue='model', aspect=1.61)
  ax.figure.savefig(f'{clargs.output_dir}/f1VSmse_scatter.png')

  for m in df['model'].unique():
    dff = df[df['model']==m]
    print(m)
    ax = seaborn.relplot(kind='line', data=dff.dropna(subset=['test_f1']), x='compression_factor', y='test_f1', hue='batch_size')
    ax.figure.suptitle(f'{m}: f1 VS compression factor')
    ax.figure.savefig(f'{clargs.output_dir}/{m}_f1VScompression_factor_line.png')
    ax = seaborn.relplot(kind='line', data=dff.dropna(subset=['mse/test']), x='compression_factor', y='mse/test', hue='batch_size')
    ax.figure.suptitle(f'{m}: Mse VS compression factor')
    ax.figure.savefig(f'{clargs.output_dir}/{m}_mseVScompression_factor_line.png')
    simple_table(dff, f'{clargs.output_dir}/{m}_summary_table')

  #cell_hover = {  # for row hover use <tr> instead of <td>
  #            'selector': 'td:hover',
  #                'props': [('background-color', '#ffffb3')]
  #                }
  #index_names = {
  #            'selector': '.index_name',
  #                'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
  #                }
  #headers = {
  #            'selector': 'th:not(.index_name)',
  #                'props': 'background-color: #eeeeee; color: #333333;'
  #                }

  #def html_img(path):
  #    if os.path.splitext(path)[1][1:] == 'png':
  #      return f'<a href="{path}"><img class="zoom" src="{path}" width="50"></a>'
  #    if os.path.splitext(path)[1][1:] == 'pdf':
  #      return f'<a href="{path}"><object class="zoom" data="{path}" width="50" height="50"></a>'
  #    return '<div style="width: 50px">:(</div>'
  #df['conf_mat'] = df['conf_mat'].apply(html_img)
  #df['umap'] = df['umap'].apply(html_img)
  #df['barplot'] = df['barplot'].apply(html_img)

  #def render_html(fname, d):
  #  with open(fname, 'w') as f:
  #    f.write('''<head>
  #    <style>
  #    .df tbody tr:nth-child(even) { background-color: lightblue; }
  #    .zoom {transition: transform .2s;}
  #    .zoom:hover{transform: scale(10);}
  #    </style>
  #    </head>
  #    <body>
  #    ''')
  #    s = d.style
  #    s.set_table_styles([cell_hover, index_names, headers])
  #    s.to_html(f, classes='df')
  #    f.write('</body>')

  #with open(f'{clargs.output_dir}/gathered_table.tex', 'w') as f:
  #  f.write('\\documentclass[12pt]{article}\n\\usepackage{booktabs}\n\\usepackage{underscore}\n\\usepackage{multirow}\n\\begin{document}\n')
  #  df.to_latex(f)
  #  f.write('\\end{decument}')
  #render_html(f'{clargs.output_dir}/gathered_table.html', df)

  #dft = df.transpose()
  #with open(f'{clargs.output_dir}/gathered_table_transpose.tex', 'w') as f:
  #  f.write('\\documentclass[12pt]{article}\n\\usepackage{booktabs}\n\\usepackage{underscore}\n\\usepackage{multirow}\n\\begin{document}\n')
  #  dft.to_latex(f)
  #  f.write('\\end{decument}')
  #render_html(f'{clargs.output_dir}/gathered_table_transpose.html', dft)


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description='Run the shape embed pipeline')
  
  parser.add_argument( 'run_folders', metavar='run_folder', nargs="+", type=str
    , help=f"The runs folders to gather results from")
  parser.add_argument( '-o', '--output-dir', metavar='OUTPUT_DIR'
    , default=f'{os.getcwd()}/gathered_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    , help=f"The OUTPUT_DIR path to use to gather results")
  parser.add_argument('-v', '--verbose', action='count', default=0
    , help="Increase verbosity level by adding more \"v\".")

  # parse command line arguments
  clargs=parser.parse_args()

  # set verbosity level
  logging.basicConfig()
  logger = logging.getLogger(__name__)
  if clargs.verbose > 1:
    logger.setLevel('DEBUG')
  elif clargs.verbose > 0:
    logger.setLevel('INFO')

  main_process(clargs, logger)
