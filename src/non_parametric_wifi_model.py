from collections import ChainMap
from pathlib import Path

import numpy as np
import pandas as pd
import pathlib
import pickle

import non_parametric_wifi_utils
import utils

# Model idea: Approximate the posterior of all training waypoints using an
# instance based non parameteric model - This allows for a very simple
# optimization free approach that makes no strong distribution assumptions
def run(mode="test", consider_multiprocessing=True, overwrite_output=False):
  models_group_name = 'non_parametric_wifi'
  overwrite_models = True
  recompute_grouped_data = not True
  # config = {
  #   'min_train_points': 10, # Ignore bssid with few observations
  #   'min_train_fns': 1, # Ignore bssid with few trajectories
  #   'delay_decay_penalty_exp_base': 0.62, # Base for bssid weight decay as a f of delay to compute the shared bssid fraction
  #   'inv_fn_count_penalty_exp': 0.1, # Exponent to give more weight to rare bssids to compute the shared bssid fraction
  #   'non_shared_penalty_start': 1.0, # Threshold below which the shared wifi fraction gets penalized in the distance calculation
  #   'non_shared_penalty_exponent': 2.2, # Exponent to penalize the non shared wifi fraction
  #   'non_shared_penalty_constant': 75, # Multiplicative constant to penalize the non shared wifi fraction
  #   'delay_decay_exp_base': 0.925, # Base for shared bssid weight decay as a f of delay
  #   'inv_fn_count_distance_exp': 0.1, # Exponent to give more weight to rare bssids to compute the weighted mean distance
  #   'unique_model_frequencies': False, # Discard bssid's with changing freqs
  #   'time_range_max_strength': 3, # Group wifi observations before and after each observation and retain the max strength
  #   'limit_train_near_waypoints': not True, # Similar to "snap to grid" - You likely want to set this to False eventually to get more granular predictions
  #   }
  config = {
  'min_train_points': 5, # Ignore bssid with few observations
  'min_train_fns': 1, # Ignore bssid with few trajectories
  'delay_decay_penalty_exp_base': 0.8, # Base for bssid weight decay as a f of delay to compute the shared bssid fraction
  'inv_fn_count_penalty_exp': 0.0, # Exponent to give more weight to rare bssids to compute the shared bssid fraction
  'non_shared_penalty_start': 1.0, # Threshold below which the shared wifi fraction gets penalized in the distance calculation
  'non_shared_penalty_exponent': 2.0, # Exponent to penalize the non shared wifi fraction
  'non_shared_penalty_constant': 50, # Multiplicative constant to penalize the non shared wifi fraction
  'delay_decay_exp_base': 0.92, # Base for shared bssid weight decay as a f of delay
  'inv_fn_count_distance_exp': 0.2, # Exponent to give more weight to rare bssids to compute the weighted mean distance
  'unique_model_frequencies': False, # Discard bssid's with changing freqs
  'time_range_max_strength': 1e-5, # Group wifi observations before and after each observation and retain the max strength
  'limit_train_near_waypoints': False # Similar to "snap to grid" - You likely want to set this to False eventually to get more granular predictions
  }
  
  debug_floor = [None, 16][0]
  debug_fn = [None, '5dd374df44333f00067aa198'][0]
  store_all_wifi_predictions = False
  store_full_wifi_predictions = not config['limit_train_near_waypoints'] # Required for the current combined optimization
  only_public_test_preds = False
  reference_submission_ext = 'non_parametric_wifi - valid - 2021-03-30 09:14:44.csv'
  bogus_test_floors_to_train_all_test_models = False
  test_override_floors = False

  data_folder = utils.get_data_folder()
  summary_path = data_folder / 'file_summary.csv'
  stratified_holdout_path = data_folder / 'holdout_ids.csv'
  leaderboard_types_path = data_folder / 'leaderboard_type.csv'
  preds_folder = data_folder.parent / 'Models' / models_group_name / (
      'predictions')
  pathlib.Path(preds_folder).mkdir(parents=True, exist_ok=True)
  if store_full_wifi_predictions:
    file_ext = models_group_name + ' - ' + mode + ' - full distances.pickle'
    full_predictions_path = preds_folder / file_ext
    
    if full_predictions_path.is_file() and (not overwrite_output):
      return
  
  reference_submission_path = data_folder / reference_submission_ext
  df = pd.read_csv(summary_path)
  holdout_df = pd.read_csv(stratified_holdout_path)
  test_waypoint_times = utils.get_test_waypoint_times(data_folder)
  test_floors = utils.get_test_floors(
    data_folder, debug_test_floor_override=test_override_floors)
  leaderboard_types = pd.read_csv(leaderboard_types_path)
  test_type_mapping = {fn: t for (fn, t) in zip(
    leaderboard_types.fn, leaderboard_types['type'])}
  reference_submission = pd.read_csv(reference_submission_path)
  
  assert store_full_wifi_predictions == (
    not config['limit_train_near_waypoints'])
  
  if bogus_test_floors_to_train_all_test_models and mode =='test':
    print("WARNING: bogus shuffling of test floors to train all floor models")
    test_floors = utils.get_test_floors(data_folder)
    site_floors = df.iloc[df.test_site.values].groupby(
      ['site_id', 'text_level']).size().reset_index()
    site_floors['level'] = [utils.TEST_FLOOR_MAPPING[t] for t in (
      site_floors.text_level)]
    site_floors['num_test_counts'] = 0
    first_floor_fns = {s: [] for s in np.unique(site_floors.site_id)}
    repeated_floor_fns = {s: [] for s in np.unique(site_floors.site_id)}
    for fn in test_floors:
      site = df.site_id[df.fn == fn].values[0]
      increment_row = np.where((site_floors.site_id == site) & (
        site_floors.level == test_floors[fn]))[0][0]
      site_floors.loc[increment_row, 'num_test_counts'] += 1
      if site_floors.num_test_counts.values[increment_row] > 1:
        repeated_floor_fns[site].append(fn)
      else:
        first_floor_fns[site].append(fn)
      
    non_visited_floor_ids = np.where(site_floors.num_test_counts == 0)[0]
    for i, non_visited_id in enumerate(non_visited_floor_ids):
      site = site_floors.site_id.values[non_visited_id]
      if repeated_floor_fns[site]:
        override_fn = repeated_floor_fns[site].pop()
      else:
        override_fn = first_floor_fns[site].pop()
      test_floors[override_fn] = site_floors.level.values[non_visited_id]
      
    # Verify that now all floors contain at least one test fn
    site_floors['num_test_counts'] = 0
    for fn in test_floors:
      site = df.site_id[df.fn == fn].values[0]
      increment_row = np.where((site_floors.site_id == site) & (
        site_floors.level == test_floors[fn]))[0][0]
      site_floors.loc[increment_row, 'num_test_counts'] += 1
      
  if debug_fn is not None:
    debug_fn_row = np.where(df.fn.values == debug_fn)[0][0]
    debug_fn_site = df.site_id.values[debug_fn_row]
    debug_fn_level = df.text_level.values[debug_fn_row]
    site_floors = df.iloc[df.test_site.values].groupby(
      ['site_id', 'text_level']).size().reset_index()
    debug_floor = np.where((site_floors.site_id == debug_fn_site) & (
      site_floors.text_level == debug_fn_level))[0][0] 
  
  use_multiprocessing = consider_multiprocessing and (
    debug_fn is None) and (debug_floor is None)
  all_outputs = non_parametric_wifi_utils.multiple_floors_train_predict(
      config, df, debug_floor, reference_submission, use_multiprocessing,
      models_group_name, mode, holdout_df, test_floors, recompute_grouped_data,
      overwrite_models, test_type_mapping, only_public_test_preds,
      test_waypoint_times, store_all_wifi_predictions,
      store_full_wifi_predictions, debug_fn)
  
  valid_preds = [r for l in [o[1] for o in all_outputs] for r in l]
  all_wifi_predictions = [r for l in [o[2] for o in all_outputs] for r in l]
  full_wifi_predictions = dict(ChainMap(*[o[3] for o in all_outputs if o[3]]))
  
  Path(preds_folder).mkdir(parents=True, exist_ok=True)
  if store_full_wifi_predictions:
    with open(full_predictions_path, 'wb') as handle:
      pickle.dump(
        full_wifi_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
  if mode == 'test':
    pass
  elif debug_floor is None:
    preds_df = pd.DataFrame(valid_preds)
    print(f"Mean validation error: {preds_df.error.values.mean():.2f}")
    
    if store_all_wifi_predictions:
      all_wifi_preds_df = pd.DataFrame(all_wifi_predictions)
      all_wifi_preds_df.sort_values(["site", "fn", "time"], inplace=True)
      preds_path = preds_folder / (
        models_group_name + ' - all wifi validation.csv')
      all_wifi_preds_df.to_csv(preds_path, index=False)
    
    holdout_unweighted = np.sqrt(preds_df.squared_error.values).mean()
    print(f"Holdout unweighted aggregate loss: {holdout_unweighted:.2f}")
