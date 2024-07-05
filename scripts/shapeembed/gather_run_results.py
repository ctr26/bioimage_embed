#! /usr/bin/env python3

import pandas as pd
import logging
import argparse
import shutil
import os
import functools
  
# define a Custom aggregation  
# function for finding total 
def keep_first_fname(series): 
  return functools.reduce(lambda x, y: y if x == 'nofile' else y, series)

def get_run_info(run):
  x = run.split('_')
  return f'{x[0]}_{x[1]}', x[2], x[4]

def main_process(clargs, logger=logging.getLogger(__name__)):
  print(clargs)
  os.makedirs(clargs.output_dir, exist_ok=True)
  dfs = []
  for d in clargs.run_folder:
    csv = f'{d}/scores_df.csv'
    #csv = f'{d}/scores_df_mean.csv'
    if not os.path.isfile(csv):
      print(f'WARNING: no {csv} found, skipping')
      continue
    
    run_name = os.path.basename(d)
    model, latent_space_sz, dataset = get_run_info(run_name)
    df = pd.read_csv(csv)
    df['model'] = model
    df['latent_space_sz'] = latent_space_sz
    df['dataset'] = dataset

    for trial in ['efd','regionprops','shapeembed', 'combined_all']:

      conf_mat = f'{trial}_confusion_matrix.png'
      if os.path.isfile(f'{d}/{conf_mat}'):
        shutil.copy(f'{d}/{conf_mat}',f'{clargs.output_dir}/{run_name}_{conf_mat}')
        df.loc[df['trial'] == trial, 'conf_mat'] = f'./{run_name}_{conf_mat}'
      else:
        df.loc[df['trial'] == trial, 'conf_mat'] = f'nofile'

      umap = f'umap_{trial}.pdf'
      if os.path.isfile(f'{d}/{umap}'):
        shutil.copy(f'{d}/{umap}',f'{clargs.output_dir}/{run_name}_{umap}')
        df.loc[df['trial'] == trial, 'umap'] = f'./{run_name}_{umap}'
      else:
        df.loc[df['trial'] == trial, 'umap'] = f'nofile'

      barplot = f'scores_barplot.pdf'
      if os.path.isfile(f'{d}/{barplot}'):
        shutil.copy(f'{d}/{barplot}',f'{clargs.output_dir}/{run_name}_{barplot}')
        df.loc[df['trial'] == trial, 'barplot'] = f'./{run_name}_{barplot}'
      else:
        df.loc[df['trial'] == trial, 'barplot'] = f'nofile'

    dfs.append(df.convert_dtypes())

  df = pd.concat(dfs)
  df = df.iloc[:, 1:] # drop first column 'unnamed' for non-mean df
  df.set_index(['dataset', 'trial', 'model', 'latent_space_sz'], inplace=True)
  df.sort_index(inplace=True)
  df = df.groupby(level=['dataset', 'trial', 'model', 'latent_space_sz']).agg({
    'test_accuracy': 'mean'
  , 'test_precision': 'mean'
  , 'test_recall': 'mean'
  , 'test_f1': 'mean'
  , 'conf_mat': keep_first_fname
  , 'umap': keep_first_fname
  , 'barplot': keep_first_fname
  })

  print('-'*80)
  print(df)
  print('-'*80)


  cell_hover = {  # for row hover use <tr> instead of <td>
              'selector': 'td:hover',
                  'props': [('background-color', '#ffffb3')]
                  }
  index_names = {
              'selector': '.index_name',
                  'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
                  }
  headers = {
              'selector': 'th:not(.index_name)',
                  'props': 'background-color: #eeeeee; color: #333333;'
                  }

  def html_img(path):
      if os.path.splitext(path)[1][1:] == 'png':
        return f'<a href="{path}"><img class="zoom" src="{path}" width="50"></a>'
      if os.path.splitext(path)[1][1:] == 'pdf':
        return f'<a href="{path}"><object class="zoom" data="{path}" width="50" height="50"></a>'
      return '<div style="width: 50px">:(</div>'
  df['conf_mat'] = df['conf_mat'].apply(html_img)
  df['umap'] = df['umap'].apply(html_img)
  df['barplot'] = df['barplot'].apply(html_img)

  def render_html(fname, d):
    with open(fname, 'w') as f:
      f.write('''<head>
      <style>
      .df tbody tr:nth-child(even) { background-color: lightblue; }
      .zoom {transition: transform .2s;}
      .zoom:hover{transform: scale(10);}
      </style>
      </head>
      <body>
      ''')
      s = d.style
      s.set_table_styles([cell_hover, index_names, headers])
      s.to_html(f, classes='df')
      f.write('</body>')

  with open(f'{clargs.output_dir}/gathered_table.tex', 'w') as f:
    f.write('\\documentclass[12pt]{article}\n\\usepackage{booktabs}\n\\usepackage{underscore}\n\\usepackage{multirow}\n\\begin{document}\n')
    df.to_latex(f)
    f.write('\\end{decument}')
  render_html(f'{clargs.output_dir}/gathered_table.html', df)

  dft = df.transpose()
  with open(f'{clargs.output_dir}/gathered_table_transpose.tex', 'w') as f:
    f.write('\\documentclass[12pt]{article}\n\\usepackage{booktabs}\n\\usepackage{underscore}\n\\usepackage{multirow}\n\\begin{document}\n')
    dft.to_latex(f)
    f.write('\\end{decument}')
  render_html(f'{clargs.output_dir}/gathered_table_transpose.html', dft)


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description='Run the shape embed pipeline')
  
  parser.add_argument( 'run_folder',  nargs="+", type=str
    , help=f"The runs folders to gather results from")
  parser.add_argument( '-o', '--output-dir', metavar='OUTPUT_DIR'
    , default=f'{os.getcwd()}/gathered_results'
    , help=f"The OUTPUT_DIR path to use to gather results")
  parser.add_argument('-v', '--verbose', action='count', default=0
    , help="Increase verbosity level by adding more \"v\".")

  # parse command line arguments
  clargs=parser.parse_args()

  # set verbosity level
  logger = logging.getLogger(__name__)
  if clargs.verbose > 2:
    logger.setLevel(logging.DEBUG)
  elif clargs.verbose > 0:
    logger.setLevel(logging.INFO)

  main_process(clargs, logger)
