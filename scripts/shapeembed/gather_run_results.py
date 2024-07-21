#! /usr/bin/env python3

import os
import shutil
import logging
import argparse
import datetime
import functools
import pandas as pd

from common_helpers import *

# define a Custom aggregation  
# function for finding total 
def keep_first_fname(series): 
  return functools.reduce(lambda x, y: y if x == 'nofile' else y, series)

def get_run_info(run):
  x = run.split('_')
  return f'{x[0]}_{x[1]}', x[2], x[4]

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

  #df = df.iloc[:, 1:] # drop first column 'unnamed' for non-mean df
  df.set_index(['dataset', 'trial', 'model', 'compression_factor', 'latent_dim', 'batch_size'], inplace=True)
  df.sort_index(inplace=True)
  df = df.groupby(level=['dataset', 'trial', 'model', 'compression_factor', 'latent_dim', 'batch_size']).agg({
    'test_accuracy': 'mean'
  , 'test_precision': 'mean'
  , 'test_recall': 'mean'
  , 'test_f1': 'mean'
  , 'conf_mat': keep_first_fname
  , 'umap': keep_first_fname
  #, 'barplot': keep_first_fname
  })

  print('-'*80)
  print(df)
  print('-'*80)
  df.to_csv(f'{clargs.output_dir}/all_scores_agg_df.csv')


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
