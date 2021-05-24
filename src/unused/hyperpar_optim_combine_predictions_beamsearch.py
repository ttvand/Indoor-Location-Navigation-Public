import datetime
import pandas as pd
from pathlib import Path
from skopt import Optimizer

import combine_predictions_beamsearch_utils
import utils


unbias_distance_predictions = True
optimize_mode = ['best_train_only', 'all_trajectories'][1]
cheat_valid_waypoints = True
experiment_path_override_ext = [
  '', 'Hyperparameter sweep 2021-04-29 20:53:08.csv'][0]
max_runs = [None, 200][0]
config_ranges = {
  'time_leak_exact_constant': [2.5, 10.0],
  'time_leak_dissimilarity_decay': [7.0, 15.0],
  'time_leak_max_penalty': [10.0, 40.0],
  'distance_pen_constant': [20.0, 50.0],
  'rel_movement_angle_constant': [0.0, 15.0],
  'abs_movement_pos_constant': [0.0, 2.0],
  'cum_abs_movement_pos_constant': [0.0, 2.0],
  'wifi_dir_constant': [0.0, 2.0],
  
  'wall_penalty_constant': [0.0, 10.0],
  
  'beam_2_width_wifi': [1, 49],
  }
fixed_config = {
  'top_distance_pos_wifi': 20,
  'weighted_pos_exponent': 4.0,
  'waypoint_weighted_wifi_penalties_mult': 0.8,
  'nn_wifi_exp': 1.5,
  'wifi_penalties_exp': 0.8,
  'nn_wifi_alpha': 0.75,
  'time_leak_delay_cutoff': 15,
  'time_leak_time_decay_constant': 20,
  'time_leak_nearby_constant': 2.0,
  'time_leak_distance_pen_limit_constant': 0.7,
  'rel_movement_pos_constant': 0.0,
  'abs_movement_angle_constant': 0.0,
  'distance_uncertainty_exponent': 1.0,
  'abs_move_uncertainty_exponent': 1.0,
  
  'inject_waypoints': not cheat_valid_waypoints,
  'off_grid_waypoint_penalty': 8.0,
  'off_grid_no_penalty_distance': 10.0,
  'dist_between_waypoints': 2.0,
  'min_distance_to_known': 3.0,
  'max_distance_to_known': 16.0,
  
  'considered_sensor_sig_keys_scale': [('z_ahrs', 0.1), ('z_magn', 1.0)],
  'top_distance_pos_sensor': 20,
  'magnetometer_penalty_constant': 0,
  
  'beam_1_width': 200,
  }
use_multiprocessing = not True
additional_grid_multiprocessing = True

mode = 'valid'
wifi_source = 'non_parametric_wifi - valid - full distances - 2021-04-16 11:47:55.pickle'
wifi_source_lgbm = 'valid_predictions_lgbm_v2.csv'
sensor_distance_source = 'distance_valid.csv'
sensor_relative_movement_source = 'relative_movement_v2_valid.csv'
sensor_absolute_movement_source = 'relative_movement_v3_valid.csv'
time_leak_source = 'valid_edge_positions_v3.csv'
sensor_signature_source = 'sensor_signature - valid - limit near waypoint - full distances - 2021-05-03 19:13:01.pickle'
sensor_uncertainties_source = 'uncertainty - valid 2021-05-07 10:48:46.csv'

wifi_ref_source = wifi_source.replace('- full distances ', '').replace(
  'pickle', 'csv')
data_folder = utils.get_data_folder()
waypoints_path = data_folder / 'train_waypoints_timed.csv'
models_folder = Path(data_folder).parent / 'Models'
wifi_preds_folder = models_folder / 'non_parametric_wifi' / 'predictions'
wifi_preds_path =  wifi_preds_folder / wifi_source
wifi_preds_lgbm_folder = models_folder / 'lgbm_wifi' / 'predictions'
wifi_preds_path_lgbm =  wifi_preds_lgbm_folder / wifi_source_lgbm
source_preds_path = wifi_preds_folder / wifi_ref_source
sensor_distance_folder = models_folder / 'sensor_distance' / 'predictions'
sensor_distance_path =  sensor_distance_folder / sensor_distance_source
sensor_rel_movement_folder = models_folder / 'sensor_relative_movement' / (
  'predictions')
sensor_abs_movement_folder = models_folder / 'sensor_absolute_movement' / (
  'predictions')
sensor_rel_movement_path =  sensor_rel_movement_folder / (
  sensor_relative_movement_source)
sensor_abs_movement_path =  sensor_abs_movement_folder / (
  sensor_absolute_movement_source)
time_leak_source_path = data_folder / time_leak_source
leaderboard_types_path = data_folder / 'leaderboard_type.csv'
sensor_signature_folder = models_folder / 'sensor_signature' / (
  'predictions')
sensor_signature_path =  sensor_signature_folder / (
  sensor_signature_source)
correct_sensor_preds_folder = models_folder / 'correct_sensor_preds' / (
  'predictions')
sensor_uncertainties_path =  correct_sensor_preds_folder / (
  sensor_uncertainties_source)
sensor_segment_stats_source = data_folder / 'sensor_data' / 'meta.csv'
walls_folder = data_folder / 'stashed_walls_intersection_count'
waypoints_folder = data_folder / 'stashed_floor_additional_waypoints'

experiments_folder = Path(data_folder).parent / 'Combined predictions' / (
  'Hyperparameter sweeps')

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
recompute_waypoints = fixed_config['inject_waypoints'] and (
  'min_distance_to_known' in config_ranges or (
    'max_distance_to_known' in config_ranges))

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
  fixed_config['beam_2_width_abs_movement'] = 50 - config[
    'beam_2_width_wifi']
  config.update(fixed_config)
  
  # Load the raw data upon changing the data mode
  if not 'wifi_preds_flat' in locals() or (
      not 'floor_waypoint_wifi_distances_order' in locals()) or (
        recompute_waypoints):
    (loaded_mode, orig_source_preds, source_preds, sites, floors,
     unique_floor_waypoints, floor_waypoint_rel_pos_distances,
     floor_waypoint_wifi_distances, floor_waypoint_wifi_distances_order,
     leaderboard_types, time_leaks, wifi_preds_flat, wifi_preds_lgbm_flat,
     original_preds, distance_preds, relative_movement_preds,
     absolute_movement_preds, sensor_preds_uncertainties, sensor_segment_stats,
     source_actual, fn_ids,
     w) = combine_predictions_beamsearch_utils.preprocess(
       config, mode, wifi_preds_path, wifi_preds_path_lgbm, source_preds_path,
       True, sensor_distance_path, sensor_rel_movement_path,
       sensor_abs_movement_path, time_leak_source_path, waypoints_path,
       leaderboard_types_path, cheat_valid_waypoints,
       sensor_uncertainties_path, sensor_segment_stats_source,
       waypoints_folder, additional_grid_multiprocessing, False)
       
  if not 'sensor_signature_dist_flat' in locals() or (
    recompute_waypoints):
    (sensor_signature_dist_flat, floor_waypoint_sensor_distances_order,
     loaded_mode_signature) = (
       combine_predictions_beamsearch_utils.preprocess_sensor_signature(
         config, mode, sensor_signature_path, sites, floors,
         unique_floor_waypoints))
  
  optimized_predictions, _ = (
    combine_predictions_beamsearch_utils.combined_predictions_all_floors(
      mode, config, use_multiprocessing, distance_preds,
      relative_movement_preds, absolute_movement_preds,
      sensor_preds_uncertainties, source_preds, original_preds, source_actual,
      sensor_segment_stats, fn_ids, sites, floors, time_leaks, wifi_preds_flat,
      wifi_preds_lgbm_flat, unique_floor_waypoints,
      floor_waypoint_rel_pos_distances, floor_waypoint_wifi_distances,
      floor_waypoint_wifi_distances_order, sensor_signature_dist_flat,
      floor_waypoint_sensor_distances_order, leaderboard_types=None,
      ignore_private_test=False, debug_fn=None,
      drop_mislabeled_fn_list_valid=[], w=w, walls_folder=walls_folder,
      unbias_distance_predictions=unbias_distance_predictions, verbose=False))
  
  err = optimized_predictions.after_optim_error.values
  mean_validation_error = err.mean()
  best_opt_err = utils.get_best_opt_error(optimized_predictions)
  tr_mask = optimized_predictions.all_targets_on_waypoints.values
  tr_traj_opt_error = err[tr_mask].mean()
  tr_best_opt_error = best_opt_err[tr_mask].mean()
  non_tr_traj_opt_error = None if cheat_valid_waypoints else (
    err[~tr_mask].mean())
  config['mean_train_trajectory_error'] = tr_traj_opt_error
  config['mean_train_trajectory_best_opt_error'] = tr_best_opt_error
  config['mean_non_train_trajectory_error'] = non_tr_traj_opt_error
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
  
  if optimize_mode == 'best_train_only':
    opt.tell([run_ids], [tr_best_opt_error])
  else:
    opt.tell([run_ids], [mean_validation_error])
  run_id += 1