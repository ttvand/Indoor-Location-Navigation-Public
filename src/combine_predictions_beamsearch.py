import numpy as np
import pandas as pd
import pathlib
from pathlib import Path

import combine_predictions_beamsearch_utils
import utils

def run(mode, grid_type, consider_multiprocessing):
  print(f"Optimizing predictions for grid type {grid_type}")
  store_valid_submission = True
  store_extended_test = True
  debug_fn = [None, '58279d6ab8c2213722f2ef6b'][0]
  extensive_search = True
  additional_grid_multiprocessing = consider_multiprocessing
  consider_ignore_private_test = False
  grid_mode = ["standard", "dense"][int(grid_type == "dense_inner")]
  grid_settings = {
    "standard": {
      "min_distance_to_known": 3.0,
      "wall_point_distance_multiplier": 0.4,
      "inner_point_distance_multiplier": 0.7,
      },
    "dense": {
      "min_distance_to_known": 1.5,
      "wall_point_distance_multiplier": 0.2,
      "inner_point_distance_multiplier": 0.35,
      },
    }
  
  grid_version = [3, 4][int(grid_type != "walls_only_old")] # V3: no inner grid fill; V4: inner grid fill + fixes
  cheat_valid_waypoints = not True
  
  config = {
    'top_distance_pos_wifi': 20,
    'weighted_pos_exponent': 4,
    'waypoint_weighted_wifi_penalties_mult': 0.8,
    'nn_wifi_exp': 1.5,
    'wifi_penalties_exp': 0.8,
    'nn_wifi_alpha': 0.75, # LGBM blend best position guess
    'time_leak_delay_cutoff': 15,
    'time_leak_time_decay_constant': 20,
    'time_leak_nearby_constant': 2,
    'time_leak_exact_constant': 5,
    'time_leak_distance_pen_limit_constant': 0.7,
    'time_leak_dissimilarity_decay': 15,
    'time_leak_max_penalty': 30,
    'distance_pen_constant': 30,
    'rel_movement_pos_constant': 0, # Angle penalties are better!
    'rel_movement_angle_constant': 9,
    'abs_movement_pos_constant': 1.5,
    'cum_abs_movement_pos_constant': 1.0,
    'abs_movement_angle_constant': 0, # Position penalties are better!
    'distance_uncertainty_exponent': 1.0,
    'abs_move_uncertainty_exponent': 1.0,
    'wifi_dir_constant': 0.5,
    
    'inject_waypoints': not cheat_valid_waypoints or mode == 'test',
    'off_grid_waypoint_penalty': 8,
    'off_grid_no_penalty_distance': 10,
    'addit_grid_density_penalty': 4,
    'min_distance_to_known': grid_settings[grid_mode]["min_distance_to_known"],
    'max_distance_to_known': 30.0,
    'generate_inner_waypoints': True,
    'generate_edge_waypoints': False,
    'wall_point_distance_multiplier': grid_settings[grid_mode][
      "wall_point_distance_multiplier"],
    'inner_point_distance_multiplier': grid_settings[grid_mode][
      "inner_point_distance_multiplier"],
    
    'considered_sensor_sig_keys_scale': [('z_ahrs', 0.1), ('z_magn', 1.0)],
    'top_distance_pos_sensor': 20,
    'magnetometer_penalty_constant': 0*1.0,
  
    'wall_penalty_constant': 0*5,
  
    'beam_1_width': (4000 if grid_mode == "very_dense" else 2000) if (
      extensive_search) else 200, # Expect a small boost of 0 to 0.05 when doing extensive_search
    'beam_2_width_wifi': 20 if extensive_search else 10,
    'beam_2_width_abs_movement': 80 if extensive_search else 40,
    }
  
  unbias_distance_predictions = True
  drop_mislabeled_fn_list_valid = []
  test_override_floors = False
  use_multiprocessing = consider_multiprocessing and (
    extensive_search) and debug_fn is None and (
      grid_mode != "very_dense") # Can be used with large (> 1000) beam 1 widths
  ignore_private_test = consider_ignore_private_test and (debug_fn is None)
  
  valid_mode = mode == 'valid'
  if valid_mode:
    wifi_source = 'non_parametric_wifi - valid - full distances.pickle'
    wifi_source_lgbm = 'valid_predictions_lgbm_v2.csv'
    sensor_distance_source = 'distance_valid.csv'
    sensor_relative_movement_source = 'relative_movement_v2_valid.csv'
    sensor_absolute_movement_source = 'relative_movement_v3_valid.csv'
    sensor_uncertainties_source = 'uncertainty - valid.csv'
    time_leak_source = 'valid_edge_positions_v3.csv'
  else:
    wifi_source = 'non_parametric_wifi - test - full distances.pickle'
    wifi_source_lgbm = 'test_predictions_lgbm_v2.csv'
    sensor_distance_source = 'distance_test.csv'
    sensor_relative_movement_source = 'relative_movement_v2_test_norefit.csv'
    sensor_absolute_movement_source = 'relative_movement_v3_test_norefit.csv'
    sensor_uncertainties_source = 'uncertainty - test.csv'
    time_leak_source = 'test_edge_positions_v3.csv'
    
  wifi_ref_source = wifi_source.replace('- full distances ', '').replace(
    'pickle', 'csv')
  wifi_ref_source = wifi_source.replace(' - full distances', '').replace(
    'pickle', 'csv')
  data_folder = utils.get_data_folder()
  waypoints_path = data_folder / 'train_waypoints_timed.csv'
  models_folder = Path(data_folder).parent / 'Models'
  wifi_preds_folder = models_folder / 'non_parametric_wifi' / 'predictions'
  storage_folder = Path(data_folder).parent / 'Combined predictions'
  pathlib.Path(storage_folder).mkdir(parents=True, exist_ok=True)
  submission_path = storage_folder / (mode + ' - ' + grid_type + '.csv')
  if submission_path.is_file():
    return
  
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
  correct_sensor_preds_folder = models_folder / 'correct_sensor_preds' / (
    'predictions')
  sensor_uncertainties_path =  correct_sensor_preds_folder / (
    sensor_uncertainties_source)
  sensor_segment_stats_source = data_folder / 'sensor_data' / 'meta.csv'
  walls_folder = data_folder / 'stashed_walls_intersection_count'
  waypoints_folder = data_folder / 'stashed_floor_additional_waypoints'
  pathlib.Path(waypoints_folder).mkdir(parents=True, exist_ok=True)
  
  # Load the raw data upon changing the data mode
  (loaded_mode, orig_source_preds, source_preds, sites, floors,
   unique_floor_waypoints, floor_waypoint_rel_pos_distances,
   floor_waypoint_wifi_distances, floor_waypoint_wifi_distances_order,
   leaderboard_types, time_leaks, wifi_preds_flat, wifi_preds_lgbm_flat,
   original_preds, distance_preds, relative_movement_preds,
   absolute_movement_preds, sensor_preds_uncertainties, sensor_segment_stats,
   source_actual, fn_ids, w) = combine_predictions_beamsearch_utils.preprocess(
     config, mode, wifi_preds_path, wifi_preds_path_lgbm, source_preds_path,
     valid_mode, sensor_distance_path, sensor_rel_movement_path,
     sensor_abs_movement_path, time_leak_source_path, waypoints_path,
     leaderboard_types_path, cheat_valid_waypoints,
     sensor_uncertainties_path, sensor_segment_stats_source,
     waypoints_folder, additional_grid_multiprocessing, test_override_floors,
     grid_version)
  
  optimized_predictions, optimized_test_predictions = (
    combine_predictions_beamsearch_utils.combined_predictions_all_floors(
      mode, config, use_multiprocessing, distance_preds, relative_movement_preds,
      absolute_movement_preds, sensor_preds_uncertainties, source_preds,
      original_preds, source_actual, sensor_segment_stats, fn_ids, sites, floors,
      time_leaks, wifi_preds_flat, wifi_preds_lgbm_flat, unique_floor_waypoints,
      floor_waypoint_rel_pos_distances, floor_waypoint_wifi_distances,
      floor_waypoint_wifi_distances_order, leaderboard_types,
      ignore_private_test, debug_fn, drop_mislabeled_fn_list_valid, w,
      walls_folder, unbias_distance_predictions))
  
  if valid_mode:
    optimized_predictions.sort_values(
      ["site", "floor", "fn", "waypoint_time"], inplace=True)
    err = optimized_predictions.after_optim_error.values
    optimized_error = err.mean()
    print(f"Optimized validation error: {optimized_error:.2f}")
    if debug_fn is None:
      best_opt_err = utils.get_best_opt_error(optimized_predictions)
      tr_mask = optimized_predictions.all_targets_on_waypoints.values
      tr_traj_opt_error = err[tr_mask].mean()
      tr_best_opt_error = best_opt_err[tr_mask].mean()
      non_tr_traj_opt_error = np.nan if cheat_valid_waypoints else (
        err[~tr_mask].mean())
      print(f"Group stats: {tr_traj_opt_error:.2f} ({tr_best_opt_error:.2f});\
   {non_tr_traj_opt_error:.2f}")
  else:
    non_predicted_ids = np.where(np.abs(original_preds).sum(1) == 0)[0]
    optimized_test_predictions = pd.concat(
      [optimized_test_predictions, orig_source_preds.iloc[non_predicted_ids]])
    original_rows = np.array([np.where(
      optimized_test_predictions.site_path_timestamp.values == sps)[
        0][0] for sps in orig_source_preds.site_path_timestamp])
        
    optimized_test_predictions = optimized_test_predictions.iloc[original_rows]
    optimized_test_predictions.index = np.arange(
      optimized_test_predictions.shape[0])
    
  if (store_valid_submission or mode == 'test') and debug_fn is None:
    if valid_mode:
      optimized_predictions.to_csv(submission_path, index=False)
    else:
      optimized_test_predictions.to_csv(submission_path, index=False)
      if store_extended_test:
        submission_path_extended = storage_folder / (
          mode + ' - ' + grid_type + ' - extended.csv')
        optimized_predictions.to_csv(submission_path_extended, index=False)