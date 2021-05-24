import datetime
import numpy as np
import pandas as pd
from skopt import Optimizer

import non_parametric_wifi_utils
import utils


models_group_name = 'non_parametric_wifi'
experiment_path_override_ext = [
  '', 'Hyperparameter sweep 2021-04-29 22:05:09.csv'][1]
debug_floor = [None, 0][0]
max_runs = [None, 100][0]
config_ranges = {
  'min_train_points': [5, 12], # Ignore bssid with few observations
  'min_train_fns': [1, 2], # Ignore bssid with few trajectories
  'delay_decay_penalty_exp_base': [0.3, 0.9], # Base for bssid weight decay as a f of delay to compute the shared bssid fraction
  'inv_fn_count_penalty_exp': [0.03, 0.3], # Exponent to give more weight to rare bssids to compute the shared bssid fraction
  'non_shared_penalty_exponent': [1.8, 2.6], # Exponent to penalize the non shared wifi fraction
  'non_shared_penalty_constant': [50.0, 80.0], # Multiplicative constant to penalize the non shared wifi fraction
  'delay_decay_exp_base': [0.88, 0.95], # Base for shared bssid weight decay as a f of delay
  'inv_fn_count_distance_exp': [0.0, 0.3], # Exponent to give more weight to rare bssids to compute the weighted mean distance
  }
fixed_config = {
  'non_shared_penalty_start': 1.0, # Threshold below which the shared wifi fraction gets penalized in the distance calculation
  'unique_model_frequencies': False, # Discard bssid's with changing freqs
  'limit_train_near_waypoints': False, # Similar to "snap to grid" - You likely want to set this to False eventually to get more granular predictions
  'time_range_max_strength': 3, # Group wifi observations before and after each observation and retain the max strength
  }
use_multiprocessing = True

data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
stratified_holdout_path = data_folder / 'holdout_ids.csv'
model_folder = data_folder.parent / 'Models' / models_group_name
experiments_folder = data_folder.parent / 'Models' / models_group_name / (
    'predictions')
if not 'df' in locals() or not 'holdout_df' in locals():
  df = pd.read_csv(summary_path)
  holdout_df = pd.read_csv(stratified_holdout_path)

# Prepare the Bayesian optimizer
config_range_keys = list(config_ranges.keys())
opt_range = [tuple(config_ranges[k]) for k in config_range_keys]
opt = Optimizer(opt_range)

record_time = str(datetime.datetime.now())[:19]
this_experiment_path = experiments_folder / (
  'Hyperparameter sweep ' + record_time + '.csv')
experiment_path_override = experiments_folder / experiment_path_override_ext
experiment_path = this_experiment_path if (
  experiment_path_override_ext == '') else experiment_path_override

if experiment_path.is_file():
  print('\nBayesian fit to earlier experiments')
  config_results = pd.read_csv(experiment_path)
  run_id = config_results.shape[0]
  suggested = config_results.iloc[:, :-(1+len(fixed_config))].values.tolist()
  target_scores = (config_results.iloc[:, -1].values).tolist()
  opt.tell(suggested, target_scores)
else:
  run_id = 0

while max_runs is None or run_id < max_runs:
  print(f"Experiment {run_id+1} - {str(datetime.datetime.now())[:19]}")
  run_ids = opt.ask(1)[0]

  config = {config_range_keys[i]: v for i, v in enumerate(run_ids)}
  config.update(fixed_config)
  
  all_outputs = non_parametric_wifi_utils.multiple_floors_train_predict(
      config, df, debug_floor, None, use_multiprocessing,
      models_group_name, 'valid', holdout_df, None, False, True, None, True,
      None, False, False, verbose=False)
  
  valid_pred_errors = np.array([r['error'] for l in [
    o[1] for o in all_outputs] for r in l])
  mean_validation_error = valid_pred_errors.mean()
  config['mean_validation_error'] = mean_validation_error
  
  # Append the experiment score
  this_config_results = pd.DataFrame([config])
  if experiment_path.is_file():
    previous_config_results = pd.read_csv(experiment_path)
    combined_config_results = pd.concat(
      [previous_config_results, this_config_results], 0)
  else:
    combined_config_results = this_config_results
  combined_config_results.to_csv(experiment_path, index=False)
  
  opt.tell([run_ids], [mean_validation_error])
  run_id += 1