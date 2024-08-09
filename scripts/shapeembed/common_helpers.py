import re
import os
import glob
import types
import logging

def compressed_n_features(dist_mat_size, comp_fact):
  return dist_mat_size*(dist_mat_size-1)//(2**comp_fact)

def model_str(params):
  s = f'{params.model_name}'
  if hasattr(params, 'model_args'):
    s += f"-{'_'.join([f'{k}{v}' for k, v in vars(params.model_args).items()])}"
  return s

def job_str(params):
  return f"{params.dataset.name}-{model_str(params)}-{params.compression_factor}-{params.latent_dim}-{params.batch_size}"

def job_str_re():
  return re.compile("(.*?)-(.*?)(-beta(\d+))*?-(\d+)-(\d+)-(\d+)")

def params_from_job_str(jobstr):
  raw_match = job_str_re().match(jobstr)
  if not raw_match: return None
  dataset_name, model_name, beta_arg, beta, cf, ld, bs = raw_match.groups()
  ps = types.SimpleNamespace()
  ps.dataset = types.SimpleNamespace(name=dataset_name)
  ps.model_name = model_name
  if beta_arg and 'beta' in beta_arg and beta:
    ps.model_args = types.SimpleNamespace()
    ps.model_args.beta = float(beta)
  ps.compression_factor = int(cf)
  ps.latent_dim = int(ld)
  ps.batch_size = int(bs)
  return ps

def find_existing_run_scores(dirname, logger=logging.getLogger(__name__)):
  ps = []
  for f in glob.glob(f'{dirname}/*-shapeembed-score_df.csv'):
    p = params_from_job_str(os.path.basename(f)[:-24])
    p.csv_file = f
    ps.append(p)
  return ps
