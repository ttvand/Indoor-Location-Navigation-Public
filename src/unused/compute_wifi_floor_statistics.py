import multiprocessing as mp

import numpy as np
import pandas as pd

import non_parametric_wifi_utils
import utils

# Model idea: Approximate the posterior of all training waypoints using an
# instance based non parameteric model - This allows for a very simple
# optimization free approach that makes no strong distribution assumptions
mode = ['valid', 'test'][1]
use_multiprocessing = not True
debug_fn = [None, '04029880763600640a0cf42c'][1]
verbose = True
debug_site = [12, None][1]
models_group_name = 'non_parametric_wifi'
# config = {
#   'min_train_points': 10, # Ignore bssid with few observations
#   'min_train_fns': 2, # Ignore bssid with few trajectories
#   'delay_decay_penalty_exp_base': 0.7, # Base for bssid weight decay as a f of delay to compute the shared bssid fraction
#   'inv_fn_count_penalty_exp': 0.2, # Exponent to give more weight to rare bssids to compute the shared bssid fraction
#   'non_shared_penalty_start': 1.0, # Threshold below which the shared wifi fraction gets penalized in the distance calculation
#   'non_shared_penalty_exponent': 2.5, # Exponent to penalize the non shared wifi fraction
#   'non_shared_penalty_constant': 65, # Multiplicative constant to penalize the non shared wifi fraction
#   'delay_decay_exp_base': 0.94, # Base for shared bssid weight decay as a f of delay
#   'inv_fn_count_distance_exp': 0.2, # Exponent to give more weight to rare bssids to compute the weighted mean distance
#   'unique_model_frequencies': False, # Discard bssid's with changing freqs
#   'time_range_max_strength': 3,
#   'limit_train_near_waypoints': True, # Similar to "snap to grid" - You likely want to set this to False eventually to get more granular predictions
#   }
config = {
  'min_train_points': 10, # Ignore bssid with few observations
  'min_train_fns': 2, # Ignore bssid with few trajectories
  'delay_decay_penalty_exp_base': 0.62, # Base for bssid weight decay as a f of delay to compute the shared bssid fraction
  'inv_fn_count_penalty_exp': 0.1, # Exponent to give more weight to rare bssids to compute the shared bssid fraction
  'non_shared_penalty_start': 1.0, # Threshold below which the shared wifi fraction gets penalized in the distance calculation
  'non_shared_penalty_exponent': 2.2, # Exponent to penalize the non shared wifi fraction
  'non_shared_penalty_constant': 75, # Multiplicative constant to penalize the non shared wifi fraction
  'delay_decay_exp_base': 0.925, # Base for shared bssid weight decay as a f of delay
  'inv_fn_count_distance_exp': 0.1, # Exponent to give more weight to rare bssids to compute the weighted mean distance
  'unique_model_frequencies': False, # Discard bssid's with changing freqs
  'time_range_max_strength': 3, # Group wifi observations before and after each observation and retain the max strength
  'limit_train_near_waypoints': True, # Similar to "snap to grid" - You likely want to set this to False eventually to get more granular predictions
  }

data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
preds_folder = data_folder.parent / 'Models' / models_group_name / (
    'predictions')
save_path = preds_folder / (mode + '_floor_pred_distances_v2.csv')
model_folder = data_folder.parent / 'Models' / models_group_name
stratified_holdout_path = data_folder / 'holdout_ids.csv'
if not 'test_floors' in locals() or (
    not 'valid_floors' in locals()):
  df = pd.read_csv(summary_path)
  test_floors = utils.get_test_floors(data_folder)
  holdout_df = pd.read_csv(stratified_holdout_path)
  assert np.all(df.fn.values[:holdout_df.shape[0]] == holdout_df.fn.values)
  df.loc[np.arange(holdout_df.shape[0]), 'mode'] = holdout_df['mode'].values
  valid_rows = np.where(
    holdout_df.test_site.values & (holdout_df['mode'].values == 'valid'))[0]
  valid_fns = holdout_df.fn.values[valid_rows]
  valid_floor_text_vals = holdout_df.text_level[valid_rows].values
  valid_floor_vals = np.array([utils.TEST_FLOOR_MAPPING[tf] for tf in (
    valid_floor_text_vals)])
  valid_floors = {k: v for (k, v) in zip(valid_fns, valid_floor_vals)}

analysis_floors = test_floors if mode =='test' else valid_floors

sites_df = df.iloc[df.test_site.values].groupby(
  ['site_id']).size().reset_index()
if debug_site is not None:
  sites_df = sites_df.iloc[debug_site:(debug_site+1)]
sites = sites_df.site_id.values

if use_multiprocessing:
  with mp.Pool(processes=mp.cpu_count()-1) as pool:
    site_ids = np.arange(sites.size)
    results = [pool.apply_async(
      non_parametric_wifi_utils.get_all_floor_preds, args=(
        mode, config, s, i, data_folder, model_folder, df,
        analysis_floors, debug_fn, verbose)) for (s, i) in zip(
          sites, site_ids)]
    all_outputs = [p.get() for p in results]
else:
  all_outputs = []
  for i, site in enumerate(sites):
    all_outputs.append(
      non_parametric_wifi_utils.get_all_floor_preds(
        mode, config, site, i, data_folder, model_folder, df, analysis_floors,
        debug_fn, verbose))

combined_preds = pd.concat(all_outputs)
combined_preds.to_csv(save_path, index=False)