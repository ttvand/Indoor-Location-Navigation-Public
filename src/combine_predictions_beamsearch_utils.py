import copy
import gc
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
import psutil

from shapely.errors import TopologicalError

import utils
from hallways.calc_old import generate_waypoints_3
from hallways.calc import generate_waypoints_4
from hallways.calc_public import get_waypoints_by_hand


def generate_floor_waypoints(
    config, mode, site, floor, waypoints, waypoints_folder, grid_version,
    add_test_grid_public_calc=False):
  floor_waypoint_ids = np.where((
    waypoints.site_id.values == site) & (
      waypoints.text_level.values == floor))[0]
  floor_waypoints = waypoints.iloc[floor_waypoint_ids]
  waypoint_locations = floor_waypoints[['x_waypoint', 'y_waypoint']].values
  unique_waypoints, train_waypoint_counts = np.unique(
    waypoint_locations, axis=0, return_counts=True)
  
  if add_test_grid_public_calc:
    public_hand_labeled = get_waypoints_by_hand(site=site, floor=floor)
    num_hand_labeled = public_hand_labeled.shape[0]
    if num_hand_labeled > 0:
      unique_waypoints = np.concatenate([
        unique_waypoints, public_hand_labeled], 0)
  
  add_public_ext = '_add_public' if add_test_grid_public_calc else ''

  if config['inject_waypoints']:
    waypoints_path = waypoints_folder / (
      mode + add_public_ext + '_' + site + '_' + str(floor) + '_' + str(
        grid_version) + '_' + str(
          float(config['min_distance_to_known'])) + '_' + str(
            float(config['max_distance_to_known'])) + '_' + str(
              config['generate_inner_waypoints']) + '_' + str(
                config['generate_edge_waypoints']) + '_' + str(
                  float(config['wall_point_distance_multiplier'])) + '_' + str(
                    float(
                      config['inner_point_distance_multiplier'])) +'.pickle')
    if not waypoints_path.is_file():
      grid_f = generate_waypoints_3 if grid_version == 3 else (
        generate_waypoints_4)
      generated_waypoints = grid_f(
          site=site, floor=floor, known_waypoints=unique_waypoints,
          min_distance_to_known=config['min_distance_to_known'],
          max_distance_to_known=config['max_distance_to_known'],
          generate_inner_waypoints=config['generate_inner_waypoints'],
          generate_edge_waypoints=config['generate_edge_waypoints'],
          wall_point_distance_multiplier=config[
            'wall_point_distance_multiplier'],
          inner_point_distance_multiplier=config[
            'wall_point_distance_multiplier'],
          )
      with open(waypoints_path, 'wb') as handle:
        pickle.dump(
          generated_waypoints, handle, protocol=pickle.HIGHEST_PROTOCOL)

def preprocess(
    config, mode, wifi_preds_path, source_preds_path, valid_mode,
    sensor_distance_path, sensor_rel_movement_path, sensor_abs_movement_path,
    time_leak_source_path, waypoints_path, leaderboard_types_path,
    cheat_valid_waypoints, sensor_uncertainties_path,
    sensor_segment_stats_source, waypoints_folder,
    additional_grid_multiprocessing, test_override_floors, grid_version,
    add_test_grid_public_calc=False):
  # Wifi data
  with open(wifi_preds_path, 'rb') as f:
    wifi_preds = pickle.load(f)
    
  source_preds = pd.read_csv(source_preds_path)
  leaderboard_types = pd.read_csv(leaderboard_types_path)
  leaderboard_types = {leaderboard_types.fn.values[i]: leaderboard_types[
    'type'].values[i] for i in range(leaderboard_types.shape[0])}
  orig_source_preds = source_preds.copy()
  sensor_preds_uncertainties = pd.read_csv(sensor_uncertainties_path)
  sensor_segment_stats_wide = pd.read_csv(
    sensor_segment_stats_source, dtype={'test_type': object})
  if valid_mode:
    source_actual = source_preds[['x_actual', 'y_actual']].values
    original_preds = source_preds[['x_pred', 'y_pred']].values
  else:
    source_actual = -1*np.ones((source_preds.shape[0], 2))
    original_preds = source_preds[['x', 'y']].values
    source_preds['fn'] = [sps.split('_')[1] for sps in (
      source_preds.site_path_timestamp)]
    source_preds['waypoint_time'] = [int(sps.split('_')[2]) for sps in (
      source_preds.site_path_timestamp)]
  
  # Sensor predictions
  distance_preds = pd.read_csv(sensor_distance_path)
  distance_preds.rename(columns={
    "pred": "prediction",
    "dist": "actual",
    }, inplace=True)
  relative_movement_preds = pd.read_csv(sensor_rel_movement_path)
  relative_movement_preds.rename(columns={
    "x_pred": "prediction_x",
    "y_pred": "prediction_y",
    "x_rel": "actual_x",
    "y_rel": "actual_y",
    }, inplace=True)
  absolute_movement_preds = pd.read_csv(sensor_abs_movement_path)
  absolute_movement_preds.rename(columns={
    "prediction_x": "x_pred",
    "prediction_y": "y_pred",
    "actual_x": "x",
    "actual_y": "y",
    }, inplace=True)
  
  time_leak = pd.read_csv(time_leak_source_path)
  
  # Identify the unique waypoints for all floors
  waypoints = pd.read_csv(waypoints_path)
  if valid_mode and not cheat_valid_waypoints:
    waypoints = waypoints[waypoints['mode'] == 'train']
  if valid_mode and cheat_valid_waypoints:
    actual_train_waypoints = waypoints[waypoints['mode'] == 'train']
  sites = []
  floors = []
  unique_floor_waypoints = []
  floor_waypoint_rel_pos_distances = []
  floor_waypoint_wifi_distances = []
  floor_waypoint_wifi_distances_order = []
  wifi_preds_flat = []
  time_leaks = []
  fn_ids = []
  sensor_segment_stats = []
  last_fn_id = 0
  
  if additional_grid_multiprocessing:
    all_sites, all_floors = zip(*wifi_preds.keys())
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
      results = [pool.apply_async(
        generate_floor_waypoints, args=(
          config, mode, s, f, waypoints, waypoints_folder,
          grid_version)) for (s, f) in zip(all_sites, all_floors)]
      [p.get() for p in results]
  else:
    for floor_id, (site, floor) in enumerate(wifi_preds.keys()):
      generate_floor_waypoints(
        config, mode, site, floor, waypoints, waypoints_folder, grid_version)
  
  for site, floor in wifi_preds.keys():
    sites.append(site)
    floors.append(floor)
    numeric_floor = utils.TEST_FLOOR_MAPPING[floor]
    wifi_preds_flat.append(wifi_preds[(site, floor)])
    sensor_segment_stats.append(sensor_segment_stats_wide[
      (sensor_segment_stats_wide.level == numeric_floor) & (
      sensor_segment_stats_wide.site == site)])
    floor_waypoint_ids = np.where((
      waypoints.site_id.values == site) & (
        waypoints.text_level.values == floor))[0]
    floor_waypoints = waypoints.iloc[floor_waypoint_ids]
    waypoint_locations = floor_waypoints[['x_waypoint', 'y_waypoint']].values
    unique_waypoints, train_waypoint_counts = np.unique(
      waypoint_locations, axis=0, return_counts=True)
    if valid_mode and cheat_valid_waypoints:
      actual_floor_waypoint_ids = np.where((
        actual_train_waypoints.site_id.values == site) & (
          actual_train_waypoints.text_level.values == floor))[0]
      actual_floor_train_waypoints = actual_train_waypoints.iloc[
        actual_floor_waypoint_ids][['x_waypoint', 'y_waypoint']].values
      unique_wp_check, double_train_waypoint_counts = np.unique(
        np.concatenate([waypoint_locations, actual_floor_train_waypoints]),
        axis=0, return_counts=True)
      assert np.all(unique_wp_check == unique_waypoints)
      train_waypoint_counts = double_train_waypoint_counts-(
        train_waypoint_counts)

    is_train_waypoint = np.ones_like(unique_waypoints[:, 0], dtype=np.bool)
    nearest_known_distances = np.zeros_like(
      is_train_waypoint, dtype=np.float32)
    addit_grid_densities = np.zeros_like(
      is_train_waypoint, dtype=np.float32)
    waypoint_types = ['train' if c > 0 else 'cheat_valid' for c in (
      train_waypoint_counts)]
    
    assert not add_test_grid_public_calc  # disallowed by Kaggle
    # if mode == 'test' and add_test_grid_public_calc:
    #   public_hand_labeled = get_waypoints_by_hand(site=site, floor=floor)
      
    #   num_hand_labeled = public_hand_labeled.shape[0]
    #   if num_hand_labeled > 0:
    #     unique_waypoints = np.concatenate([
    #       unique_waypoints, public_hand_labeled], 0)
    #     is_train_waypoint = np.ones_like(unique_waypoints[:, 0], dtype=np.bool)
    #     train_waypoint_counts = np.concatenate([
    #       train_waypoint_counts, np.zeros(num_hand_labeled)])
    #     waypoint_types += ['hand_labeled']*num_hand_labeled

    if config['inject_waypoints']:
      try:
        add_public_ext = '_add_public' if add_test_grid_public_calc else ''
        waypoints_path = waypoints_folder / (
          mode + add_public_ext + '_' + site + '_' + str(floor) + '_' + str(
            grid_version) + '_' + str(
              float(config['min_distance_to_known'])) + '_' + str(
                float(config['max_distance_to_known'])) + '_' + str(
                  config['generate_inner_waypoints']) + '_' + str(
                    config['generate_edge_waypoints']) + '_' + str(
                      float(config[
                        'wall_point_distance_multiplier'])) + '_' + str(float(
                          config[
                            'inner_point_distance_multiplier'])) +'.pickle')
        assert waypoints_path.is_file()
        with open(waypoints_path, 'rb') as f:
          generated_waypoints_types = pickle.load(f)
          
        generated_waypoints = np.concatenate(generated_waypoints_types)
        
        known_waypoints = np.copy(unique_waypoints)
        unique_waypoints = np.concatenate(
            (unique_waypoints, generated_waypoints))
        num_generated = generated_waypoints.shape[0]
        if num_generated > 0:
          num_wall = generated_waypoints_types[0].shape[0]
          num_inner = num_generated-num_wall
          is_train_waypoint = np.concatenate(
              (is_train_waypoint, np.zeros_like(
                generated_waypoints[:, 0], dtype=np.bool)))
          train_waypoint_counts = np.concatenate(
              (train_waypoint_counts, np.zeros(
                num_generated, dtype=train_waypoint_counts.dtype)))
          waypoint_types += (['grid_wall']*num_wall + ['grid_inner']*num_inner)
          addit_grid_distances = np.sqrt(
            (known_waypoints[:, :1] - np.expand_dims(
              generated_waypoints[:, 0], 0))**2 + (
                known_waypoints[:, 1:] - np.expand_dims(
                  generated_waypoints[:, 1], 0))**2)
          addit_grid_nearest_distances = addit_grid_distances.min(0)
          nearest_known_distances = np.concatenate(
              (nearest_known_distances, addit_grid_nearest_distances))
          
          grid_grid_distances = np.sqrt(
            (generated_waypoints[:, :1] - np.expand_dims(
              generated_waypoints[:, 0], 0))**2 + (
                generated_waypoints[:, 1:] - np.expand_dims(
                  generated_waypoints[:, 1], 0))**2)
          np.fill_diagonal(grid_grid_distances, 1e9)
          grid_grid_distances[grid_grid_distances > 5] = 1e9
          grid_grid_distances = np.maximum(grid_grid_distances, 1)
          generated_grid_densities = (1/grid_grid_distances).sum(0)
          addit_grid_densities = np.concatenate(
              (addit_grid_densities, generated_grid_densities))
        
        # if site == '5dbc1d84c1eb61796cf7c010' and floor == "F6":
        #   import pdb; pdb.set_trace()
        #   x=1
        
      except TopologicalError as exc:
        print(f"Floor plan error on site {site} floor {floor}:", exc)
        pass

    unique_floor_waypoints.append(
      (unique_waypoints, is_train_waypoint, np.array(waypoint_types),
       train_waypoint_counts, nearest_known_distances, addit_grid_densities))
    
    x_diff = unique_waypoints[:, :1] - np.transpose(unique_waypoints[:, :1])
    y_diff = unique_waypoints[:, 1:] - np.transpose(unique_waypoints[:, 1:])
    distances = np.sqrt(x_diff**2 + y_diff**2)
    angles = np.angle(x_diff + 1j*y_diff)
    floor_waypoint_rel_pos_distances.append(np.stack([
      x_diff, y_diff, distances, angles]))
    
    time_leaks.append(time_leak[(time_leak.actual_floor == numeric_floor) & (
      time_leak.site == site)])
    
    if wifi_preds[(site, floor)] is not None:
      wifi_positions = wifi_preds[(site, floor)].values[:, :2]
      x_diff = wifi_positions[:, :1] - np.transpose(unique_waypoints[:, :1])
      y_diff = wifi_positions[:, 1:] - np.transpose(unique_waypoints[:, 1:])
      distances = np.sqrt(x_diff**2 + y_diff**2)
      
      floor_waypoint_wifi_distances.append(distances)
      floor_waypoint_wifi_distances_order.append(np.argsort(
        distances, 0))
      
      fns_floor = np.unique(
        [c.split('_')[0] for c in wifi_preds[(site, floor)].columns[2:]])
      fns_floor = fns_floor.astype(object)
      fn_ids.append({fn: i+last_fn_id for i, fn in enumerate(fns_floor)})
      last_fn_id += fns_floor.size
    else:
      floor_waypoint_wifi_distances.append(None)
      floor_waypoint_wifi_distances_order.append(None)
      fn_ids.append(None)
  
  w = None

  loaded_mode = mode
  
  # import pdb; pdb.set_trace()
  return (loaded_mode, orig_source_preds, source_preds, sites, floors,
          unique_floor_waypoints, floor_waypoint_rel_pos_distances,
          floor_waypoint_wifi_distances, floor_waypoint_wifi_distances_order,
          leaderboard_types, time_leaks, wifi_preds_flat, original_preds,
          distance_preds, relative_movement_preds, absolute_movement_preds,
          sensor_preds_uncertainties, sensor_segment_stats, source_actual,
          fn_ids, w)


def preprocess_sensor_signature(
    config, mode, sensor_signature_path, sites, floors,
    unique_floor_waypoints):
  load_sensor = config['magnetometer_penalty_constant'] > 0
  if load_sensor:
    with open(sensor_signature_path, 'rb') as f:
      sensor_signature_distances = pickle.load(f)
  else:
    sensor_signature_distances = {}
    
  sensor_signature_dist_flat = []
  floor_waypoint_sensor_distances_order = []
  for i, (s, f) in enumerate(zip(sites, floors)):
    k = (s, f)
    key_present = k in sensor_signature_distances
    if key_present:
      sensor_dist = sensor_signature_distances[k]
      sensor_keys = list(sensor_dist.keys())
      sensor_signature_dist_flat.append(sensor_dist)
    else:
      sensor_signature_dist_flat.append(None)
    
    if key_present and isinstance(sensor_dist[sensor_keys[0]], pd.DataFrame):
      unique_waypoints = unique_floor_waypoints[i][0]
      sensor_positions = sensor_dist[sensor_keys[0]].values[:, :2]
      x_diff = sensor_positions[:, :1] - np.transpose(unique_waypoints[:, :1])
      y_diff = sensor_positions[:, 1:] - np.transpose(unique_waypoints[:, 1:])
      distances = np.sqrt(x_diff**2 + y_diff**2)
      distance_order = np.argsort(distances, 0)
      ordered_distanced = np.sort(distances, 0)
      
      floor_waypoint_sensor_distances_order.append((
        distances, distance_order, ordered_distanced))
    else:
      floor_waypoint_sensor_distances_order.append((None, None, None))
    
  del sensor_signature_distances
  gc.collect()

  loaded_mode_signature = mode
  
  return (sensor_signature_dist_flat, floor_waypoint_sensor_distances_order,
          loaded_mode_signature)

def get_pred_penalties(penalties, trace):
  num_preds = len(penalties)
  num_penalties = len(penalties[0])-1
  pred_penalties = np.zeros((num_preds, num_penalties))
  for i in range(num_preds):
    target_id = np.where(
      np.all(penalties[i][0] == np.expand_dims(trace[:i+1], 0), 1))[0][0]
    for j in range(num_penalties):
      pred_penalties[i, j] = penalties[i][j+1][target_id]
    
  return pred_penalties

def time_leak_edge(
    delay_edge, time_leak_delay_cutoff, time_leak_time_decay_constant,
    time_leak_exact_constant, time_leak_nearby_constant,
    time_leak_distance_pen_limit_constant, dissimilarity_constant,
    dissimilarity, edge_pos, candidate_pos):
  delay_edge = delay_edge.values[0]/1000
  assert delay_edge > 0
  edge_pos = edge_pos.values
  dissimilarity = dissimilarity.values
  if delay_edge < time_leak_delay_cutoff and not np.any(np.isnan(edge_pos)):
    time_multiplier = 1/(1+delay_edge/time_leak_time_decay_constant)
    distances = np.sqrt(((candidate_pos - edge_pos)**2).sum(1))
    distance_pen_limit = np.log2(delay_edge+1)*(
      time_leak_distance_pen_limit_constant)
    dissimilarity_multiplier = 1 if (
      dissimilarity <= dissimilarity_constant) else max(0, 1 - 1/(
        dissimilarity_constant**1.5)*(
          dissimilarity-dissimilarity_constant)**1.5)
    time_leak_penalties = dissimilarity_multiplier * time_multiplier *(
      (distances > 1e-3) * time_leak_exact_constant + np.maximum(
        0, time_leak_nearby_constant*(
          distances/distance_pen_limit - 1))**2)
    if np.any(np.isnan(time_leak_penalties)):
      raise ValueError("Nan in time leak penalty")
  else:
    time_leak_penalties = np.zeros_like(candidate_pos[:, 0])
    
  return time_leak_penalties


def align_predictions(
    config, wifi_positions, wifi_times, wifi_fn_distances, sensor_positions,
    sensor_times, sensor_fn_distances, waypoint_times, floor_waypoints,
    off_grid_penalties, addit_grid_density_penalties,
    floor_waypoint_rel_pos_distances, floor_waypoint_wifi_distances,
    floor_waypoint_wifi_distances_order, floor_waypoint_sensor_distances_order,
    fn_distance_preds, fn_relative_movement_preds, fn_absolute_movement_preds,
    fn_sensor_preds_uncertainties, fn_time_leak, actual_waypoints, valid_mode,
    w, walls_count, unbias_distance_predictions):
  
  top_distance_pos_wifi = config['top_distance_pos_wifi']
  weighted_pos_exponent = config['weighted_pos_exponent']
  waypoint_weighted_wifi_penalties_mult = config[
    'waypoint_weighted_wifi_penalties_mult']
  nn_wifi_exp = config['nn_wifi_exp']
  wifi_penalties_exp = config['wifi_penalties_exp']
  time_leak_delay_cutoff = config['time_leak_delay_cutoff']
  time_leak_time_decay_constant = config['time_leak_time_decay_constant']
  time_leak_nearby_constant = config['time_leak_nearby_constant']
  time_leak_exact_constant = config['time_leak_exact_constant']
  time_leak_distance_pen_limit_constant = config[
    'time_leak_distance_pen_limit_constant']
  time_leak_dissimilarity_decay = config['time_leak_dissimilarity_decay']
  time_leak_max_penalty = config['time_leak_max_penalty']
  distance_pen_constant = config['distance_pen_constant']
  # distance_laplace_smoother = config['distance_laplace_smoother']
  rel_movement_pos_constant = config['rel_movement_pos_constant']
  rel_movement_angle_constant = config['rel_movement_angle_constant']
  abs_movement_pos_constant = config['abs_movement_pos_constant']
  cum_abs_movement_pos_constant = config['cum_abs_movement_pos_constant']
  abs_movement_angle_constant = config['abs_movement_angle_constant']
  distance_uncertainty_exponent = config['distance_uncertainty_exponent']
  abs_move_uncertainty_exponent = config['abs_move_uncertainty_exponent']
  wifi_dir_constant = config['wifi_dir_constant']
  top_distance_pos_sensor = config['top_distance_pos_sensor']
  magnetometer_penalty_constant = config['magnetometer_penalty_constant']
  wall_penalty_constant = config['wall_penalty_constant']

  beam_1_width = config['beam_1_width']
  beam_2_width_wifi = config['beam_2_width_wifi']
  beam_2_width_abs_movement = config['beam_2_width_abs_movement']
  
  # Shared: compute the interpolation ids
  fn = fn_distance_preds.fn.values[0]
  site = fn_distance_preds.site.values[0]
  floor = fn_distance_preds.floor.values[0]
  num_waypoints = floor_waypoints.shape[0]
  num_pred_times = wifi_times.size
  num_pred_waypoints = waypoint_times.size
  num_times_greater_wifi = (
    np.expand_dims(waypoint_times, 1) > np.expand_dims(wifi_times, 0)).sum(1)
  low_ids_wifi = np.maximum(0, num_times_greater_wifi-1)
  high_ids_wifi = np.minimum(num_pred_times-1, num_times_greater_wifi)
  low_time_offsets = np.abs(wifi_times[low_ids_wifi] - waypoint_times)/1000
  high_time_offsets = np.abs(wifi_times[high_ids_wifi] - waypoint_times)/1000
  
  # Compute the sensor distances for all waypoints for all wifi times
  consider_sensor = config['magnetometer_penalty_constant'] > 0
  if consider_sensor:
    num_sensor_times = sensor_times.size
    num_times_greater_sensor = (
      np.expand_dims(waypoint_times, 1) > np.expand_dims(sensor_times, 0)).sum(1)
    low_ids_sensor = np.maximum(0, num_times_greater_sensor-1)
    high_ids_sensor = np.minimum(num_sensor_times-1, num_times_greater_sensor)
    near_sensor_ids_flattened = np.transpose(
      floor_waypoint_sensor_distances_order[1][
        :top_distance_pos_sensor]).flatten()
    waypoint_sensor_distances = sensor_fn_distances[
      :, near_sensor_ids_flattened].reshape((
        -1, num_waypoints, top_distance_pos_sensor, num_sensor_times)).transpose(
          (0, 3, 1, 2))
    waypoint_sensor_distances_sorted = np.sort(waypoint_sensor_distances, 3)[
      :, :, :, :top_distance_pos_wifi//4]
    time_sensor_weighted_dist = 1.5*waypoint_sensor_distances_sorted.mean(3) + (
      1.5*(waypoint_sensor_distances_sorted.min(3)))
    min_sensor_dist = time_sensor_weighted_dist.min(2, keepdims=True)
    time_sensor_penalties = time_sensor_weighted_dist - min_sensor_dist
    nn_waypoint_sensor_penalties = (time_sensor_penalties[:, low_ids_sensor] + (
        time_sensor_penalties[:, high_ids_sensor]))/2 * (
          magnetometer_penalty_constant)
    # nearest_sensor_distances = floor_waypoint_sensor_distances_order[2][0]
  else:
    nn_waypoint_sensor_penalties = np.zeros((2, 1, num_waypoints))
  
  # Compute the wifi distances for all waypoints for all wifi times
  near_wifi_ids_flattened = np.transpose(floor_waypoint_wifi_distances_order[
    :top_distance_pos_wifi]).flatten()
  waypoint_wifi_distances = wifi_fn_distances[near_wifi_ids_flattened].reshape(
    (num_waypoints, top_distance_pos_wifi, num_pred_times)).transpose(
      (2, 0, 1))
  
  # Compute the nearest waypoint seen in training for all actual waypoints
  actual_nearest_train_distances = np.zeros(num_pred_waypoints)
  actual_waypoint_ids = -1*np.ones(num_pred_waypoints, dtype=np.int32)
  nearest_waypoint_ids = -1*np.ones(num_pred_waypoints, dtype=np.int32)
  for i in range(num_pred_waypoints):
    squared_distances = (floor_waypoints[:, 0] - actual_waypoints[i, 0])**2 + (
      floor_waypoints[:, 1] - actual_waypoints[i, 1])**2
    actual_nearest_train_distances[i] = np.sqrt(squared_distances.min())
    rep_penalties = np.zeros_like(squared_distances)
    if i > 0:
      rep_penalties[nearest_waypoint_ids[i-1]] = 1e6
    nearest_waypoint_ids[i] = np.argmin(squared_distances + rep_penalties)
    if actual_nearest_train_distances[i] == 0:
      actual_waypoint_ids[i] = squared_distances.argmin()
  target_on_waypoints = actual_waypoint_ids >= 0
  # all_targets_on_waypoints = np.all(target_on_waypoints)
  
  # import pdb; pdb.set_trace()
  # x=2
  
  # Compute a weighted position using the nearest training neighbors
  # (wifi position based)
  top_ids = np.transpose(
    np.argsort(wifi_fn_distances, 0)[:top_distance_pos_wifi]).flatten()
  top_pos = wifi_positions[top_ids]
  repeated_ids = np.repeat(np.arange(num_pred_times), top_distance_pos_wifi)
  top_distances = wifi_fn_distances[(top_ids, repeated_ids)].reshape((
    num_pred_times, top_distance_pos_wifi))
  min_distance = np.expand_dims(top_distances.min(1), 1)
  pos_weights = 1/((top_distances+1e-5)/(min_distance+1e-5))**(
    weighted_pos_exponent)
  weighted_pos = (top_pos*np.expand_dims(
    pos_weights.flatten(), 1)).reshape(
      num_pred_times, top_distance_pos_wifi, 2).sum(1)/pos_weights.sum(
        1, keepdims=True)
  before_optim_preds = (weighted_pos[low_ids_wifi] + weighted_pos[
    high_ids_wifi])/2
  # before_optim_preds = 0.75*before_optim_preds + 0.25*lgbm_pos
  weighted_pos_distances = np.sqrt((
    weighted_pos[:, :1] - np.expand_dims(floor_waypoints[:, 0], 0))**2 + (
      weighted_pos[:, 1:] - np.expand_dims(floor_waypoints[:, 1], 0))**2)
  time_weighted_wifi_penalties = weighted_pos_distances - (
    weighted_pos_distances.min(1, keepdims=True))
  waypoint_weighted_wifi_penalties_nn = (
    time_weighted_wifi_penalties[low_ids_wifi] + (
      time_weighted_wifi_penalties[high_ids_wifi]))/2
  waypoint_weighted_wifi_penalties = waypoint_weighted_wifi_penalties_nn
  waypoint_weighted_wifi_penalties = waypoint_weighted_wifi_penalties ** (
    wifi_penalties_exp)
  
  # Compute the distances for all waypoints to the nearest training points
  # (waypoint position based)
  waypoint_wifi_distances_sorted = np.sort(waypoint_wifi_distances, 2)[
    :, :, :top_distance_pos_wifi//2]
  time_wifi_weighted_dist = 1.5*waypoint_wifi_distances_sorted.mean(2) + 1.5*(
    waypoint_wifi_distances_sorted.min(2))
  # time_wifi_weighted_dist = waypoint_wifi_distances.mean(2) + 2*(
  #   waypoint_wifi_distances.min(2))
  
  min_wifi_dist = time_wifi_weighted_dist.min(1, keepdims=True)
  time_wifi_penalties = time_wifi_weighted_dist - min_wifi_dist
  low_weights = np.expand_dims(1/np.maximum(2, low_time_offsets), 1)
  high_weights = np.expand_dims(1/np.maximum(2, high_time_offsets), 1)
  weight_sums = low_weights + high_weights
  nn_waypoint_wifi_penalties = (low_weights*time_wifi_penalties[
    low_ids_wifi] + (
      high_weights*time_wifi_penalties[high_ids_wifi]))/weight_sums
  time_wifi_min_dist = ((low_weights*min_wifi_dist[low_ids_wifi] + (
    high_weights*min_wifi_dist[high_ids_wifi]))/weight_sums)[:, 0]
  # wifi_time_mult_unknown = np.expand_dims(1*(1-(np.minimum(
  #   8, np.maximum(15, time_wifi_min_dist)-15))/10), 1)
  wifi_time_mult_unknown = 1
  nn_waypoint_wifi_penalties = nn_waypoint_wifi_penalties**nn_wifi_exp
  
  # Give less weight to WiFi data if it is out of sync
  wifi_mult = weight_sums.sum(1, keepdims=True)*wifi_time_mult_unknown
  waypoint_wifi_penalties = wifi_mult * np.stack([
    nn_waypoint_wifi_penalties,
    waypoint_weighted_wifi_penalties*waypoint_weighted_wifi_penalties_mult,
    ], -1).min(-1)
  
  distance_preds = fn_distance_preds.prediction.values
  if unbias_distance_predictions:
    distance_preds[0] = max(1, distance_preds[0] - 0.3147)
    distance_preds[-1] = distance_preds[-1] + 0.1577
    
  relative_movement_preds = fn_relative_movement_preds[[
    'prediction_x', 'prediction_y']].values
  absolute_movement_preds = fn_absolute_movement_preds[[
    'x_pred', 'y_pred']].values
  distance_uncertainty_preds = (
    fn_sensor_preds_uncertainties.pred_distance_uncertainty.values)
  abs_move_uncertainty_preds = (
    fn_sensor_preds_uncertainties.pred_angle_uncertainty.values)

  # Cheating to get an idea of the marginal gains
  # distance_preds = fn_distance_preds.actual.values
  # relative_movement_preds = fn_relative_movement_preds[[
  #   'actual_x', 'actual_y']].values
  # absolute_movement_preds = fn_absolute_movement_preds[['x', 'y']].values
  # distance_uncertainty_preds = (
  #   fn_sensor_preds_uncertainties.distance_uncertainty.values)
  # abs_move_uncertainty_preds = (
  #   fn_sensor_preds_uncertainties.angle_uncertainty.values)


  first_candidate_penalties = waypoint_wifi_penalties[0] + (
    nn_waypoint_sensor_penalties[1][0]) + (
      off_grid_penalties + addit_grid_density_penalties)
  candidates = np.argsort(waypoint_wifi_penalties[0])[:beam_1_width].reshape(
    -1, 1)
  
  # Hack to make sure we keep the nearest waypoint in the starting beam.
  # Not a concern for overfitting since this is only a rare issue when not
  # doing a wide search
  if valid_mode and not nearest_waypoint_ids[0] in candidates:
    candidates[-1] = nearest_waypoint_ids[0]
    if beam_1_width > 1000:
      print("VALIDATION OVERFITTING WARNING - LOOK INTO ME")
  
  candidate_penalties = first_candidate_penalties[candidates[:, 0]]
  all_penalties = [(
    candidates, waypoint_wifi_penalties[0][candidates[:, 0]],
    np.zeros_like(candidate_penalties), np.zeros_like(candidate_penalties),
    np.zeros_like(candidate_penalties), np.zeros_like(candidate_penalties),
    np.zeros_like(candidate_penalties), np.zeros_like(candidate_penalties),
    off_grid_penalties[candidates[:, 0]],
    addit_grid_density_penalties[candidates[:, 0]],
    nn_waypoint_sensor_penalties[1][0][candidates[:, 0]])]
  num_beam_2_wifi_candidates = min(num_waypoints, beam_2_width_wifi)
  num_beam_2_abs_pos_candidates = min(
    num_waypoints-num_beam_2_wifi_candidates, beam_2_width_abs_movement)
  num_beam_2_candidates = num_beam_2_wifi_candidates + (
    num_beam_2_abs_pos_candidates)
  
  # Beam search using a combined penalty of all waypoints
  for i in range(1, num_pred_waypoints):
    beam_2_wifi_candidates = np.argsort(waypoint_wifi_penalties[i])[
      :num_beam_2_wifi_candidates]
    waypoint_angles = floor_waypoint_rel_pos_distances[3][
      candidates[:, -1], :] + np.pi
    abs_move_angle = np.angle(
      absolute_movement_preds[i-1, 0] + 1j*absolute_movement_preds[i-1, 1])
    abs_angle_errors = np.abs(waypoint_angles - abs_move_angle)
    angle_errors = np.minimum(
      np.abs(abs_angle_errors), 2*np.pi - abs_angle_errors)
    rel_x_y_dist = floor_waypoint_rel_pos_distances[:3, candidates[:, -1]]
    rel_movements_x_error = np.abs(rel_x_y_dist[0] + (
        absolute_movement_preds[i-1, 0]))
    rel_movements_y_error = np.abs(rel_x_y_dist[1] + (
        absolute_movement_preds[i-1, 1]))
    rel_movements_dist_error = np.abs(rel_x_y_dist[2] - distance_preds[i-1])

    beam_2_non_repeat_wifi_penalties = np.zeros_like(rel_movements_x_error)
    beam_2_non_repeat_wifi_penalties[:, beam_2_wifi_candidates] = 1e7
    rel_move_bad_angle_penalties = 1e6 * (angle_errors > np.pi/4)  
    beam_2_abs_pos_candidates = np.argsort(
      rel_movements_x_error + rel_movements_y_error + 4*(
        rel_movements_dist_error) + (
          beam_2_non_repeat_wifi_penalties + rel_move_bad_angle_penalties),
          1)[:, :num_beam_2_abs_pos_candidates]
    
    if num_beam_2_abs_pos_candidates == 0:
      next_step_candidates = np.tile(
        beam_2_wifi_candidates, candidates.shape[0]).flatten()
    elif num_beam_2_wifi_candidates == 0:
      next_step_candidates = beam_2_abs_pos_candidates.flatten()
    else:
      next_step_candidates = np.concatenate([
        np.tile(beam_2_wifi_candidates, candidates.shape[0]).reshape(
          (-1, num_beam_2_wifi_candidates)),
        beam_2_abs_pos_candidates], 1).flatten()
    
    next_candidates = np.concatenate([
      np.repeat(candidates, num_beam_2_candidates, axis=0),
      np.expand_dims(next_step_candidates, -1)], 1)
    
    beam_2_no_cheat_penalties = np.zeros_like(next_step_candidates)
    target_id_nearest = np.where(np.all(next_candidates == np.expand_dims(
      nearest_waypoint_ids[:i+1], 0), 1))[0]
    if valid_mode and not target_id_nearest.size:
      prev_target_rows = np.where(np.all(
        next_candidates[:, :-1] == np.expand_dims(
          nearest_waypoint_ids[:i], 0), 1))[0]
      next_candidates[prev_target_rows[-1], -1] = nearest_waypoint_ids[i]
      target_id_nearest = np.where(np.all(next_candidates == np.expand_dims(
        nearest_waypoint_ids[:i+1], 0), 1))[0]
      beam_2_no_cheat_penalties[prev_target_rows[-1]] = 1e6

    # Wall penalties
    wall_penalties = np.zeros_like(beam_2_no_cheat_penalties)
    if wall_penalty_constant > 0:
      keys = list(zip(
        floor_waypoints[next_candidates[:, -2]][:, 0],
        floor_waypoints[next_candidates[:, -2]][:, 1],
        floor_waypoints[next_candidates[:, -1]][:, 0],
        floor_waypoints[next_candidates[:, -1]][:, 1],
        ))
      wall_counts = np.zeros_like(wall_penalties)
      for wall_id in range(wall_penalties.size):
        k = keys[wall_id]
        if k in walls_count:
          wall_counts[wall_id] = walls_count[k]
        else:
          wall_count = w.count_walls(*tuple([site, floor] + list(k)))
          walls_count[k] = wall_count
          wall_counts[wall_id] = wall_count

      wall_penalties = wall_penalty_constant*np.sqrt(wall_counts)

    # WiFi, self-repeat and distance penalties
    prev_penalties = np.repeat(candidate_penalties, num_beam_2_candidates)
    wifi_penalties = waypoint_wifi_penalties[i][next_candidates[:, -1]]
    if consider_sensor:
      magnetometer_abs_pos_penalties = nn_waypoint_sensor_penalties[1][i][
        next_candidates[:, -1]]
    else:
      magnetometer_abs_pos_penalties = np.zeros_like(prev_penalties)
    self_repeat_penalties = 1e6*(
      next_candidates[:, -2] == next_candidates[:, -1])
    candidate_distances = floor_waypoint_rel_pos_distances[(
      2, next_candidates[:, -2], next_candidates[:, -1])]
    # distance_penalties = distance_pen_constant*(np.abs(1-(
    #   candidate_distances+distance_laplace_smoother)/(
    #     distance_preds[i-1]+distance_laplace_smoother)))
    dist_uncertainty_mult = min(4, (
      0.12/distance_uncertainty_preds[i-1])**distance_uncertainty_exponent)
    distance_penalties = dist_uncertainty_mult*distance_pen_constant/5*np.abs(
      candidate_distances-distance_preds[i-1])/(
        1.4*(1-1/(distance_preds[i-1]**0.5+0.5)))

    # WiFi direction penalty
    wifi_vector = before_optim_preds[i] - before_optim_preds[i-1]
    wifi_vector_norm = np.sqrt((wifi_vector**2).sum())
    wifi_vector_mult = (wifi_vector_norm)**(1/2)
    rel_movements = floor_waypoints[
        next_candidates[:, -1]] - floor_waypoints[next_candidates[:, -2]]
    rel_movement_norms = np.sqrt((rel_movements**2).sum(1))
    cos_sim = (rel_movements * np.expand_dims(wifi_vector, 0)).sum(-1)/(
      (wifi_vector_norm+1e-9)*(rel_movement_norms+1e-9))
    wifi_dir_penalties = -wifi_dir_constant*wifi_vector_mult*cos_sim
        
    # Absolute movement penalty
    abs_movement_diff = np.abs(rel_movements - np.expand_dims(
      absolute_movement_preds[i-1], 0)).sum(1)
    rel_move_x = floor_waypoints[next_candidates[:, -1], :1] - floor_waypoints[
      next_candidates[:, :-1].flatten(), 0].reshape((-1, i))
    rel_move_y = floor_waypoints[next_candidates[:, -1], 1:] - floor_waypoints[
      next_candidates[:, :-1].flatten(), 1].reshape((-1, i))
    cum_abs_targets = np.flip(
      np.cumsum(np.flip(absolute_movement_preds[:i], 0), 0), 0)
    agg_error_divisor = np.flip(np.sqrt(1+np.arange(i)))
    cum_abs_errors = np.abs(rel_move_x - np.expand_dims(
      cum_abs_targets[:, 0], 0)) + np.abs(rel_move_y - np.expand_dims(
        cum_abs_targets[:, 1], 0))
    combined_cum_abs_error = (
      cum_abs_errors / np.expand_dims(agg_error_divisor, 0)).mean(1)
    absolute_movement_pos_penalties = abs_movement_pos_constant*(
      abs_movement_diff)
    cum_absolute_movement_pos_penalties = cum_abs_movement_pos_constant*(
      combined_cum_abs_error)
    
    waypoint_angles = floor_waypoint_rel_pos_distances[3][(
      next_candidates[:, -1], next_candidates[:, -2])]
    abs_angle_errors = np.abs(waypoint_angles - abs_move_angle)
    angle_errors = np.minimum(
      np.abs(abs_angle_errors), 2*np.pi - abs_angle_errors)
    exclude_half_plain_abs_penalties = 1e6 * (angle_errors > np.pi/2)
    # import pdb; pdb.set_trace()
    # if exclude_half_plain_abs_penalties[target_id_nearest] > 0:
    #   print("Excluding nearest from the correct half plane")
    
    # pred_abs_angle = np.angle(absolute_movement_preds[i-1, 0] + 1j * (
    #   absolute_movement_preds[i-1, 1]))
    # cand_angles = np.angle(rel_movements[:, 0] + 1j * rel_movements[:, 1])
    # abs_angle_error = np.abs(pred_abs_angle - cand_angles)
    # angle_error = np.minimum(
    #   np.abs(abs_angle_error), 2*np.pi - abs_angle_error)
    # absolute_movement_angle_penalties = (
    #   abs_movement_angle_constant*angle_error)
    if i == 1:
      absolute_movement_angle_penalties = np.zeros_like(prev_penalties)
    else:
      prev_rel_movements = floor_waypoints[
        candidates[:, -2]] - floor_waypoints[candidates[:, -1]]
      cand_angles = np.angle(prev_rel_movements[:, 0] + 1j * (
        prev_rel_movements[:, 1]))
      next_cand_angles = np.repeat(cand_angles, num_beam_2_candidates)
      next_rel_x = floor_waypoint_rel_pos_distances[(
        0, next_candidates[:, -2], next_candidates[:, -1])]
      next_rel_y = floor_waypoint_rel_pos_distances[(
        1, next_candidates[:, -2], next_candidates[:, -1])]
      cand_rel_rot_x = np.cos(next_cand_angles)*next_rel_x + np.sin(
        next_cand_angles)*next_rel_y
      cand_rel_rot_y = -np.sin(next_cand_angles)*next_rel_x + np.cos(
        next_cand_angles)*next_rel_y
      cand_rel_angles = np.angle(cand_rel_rot_x + 1j * cand_rel_rot_y)
      
      prev_pred_angle = np.angle(absolute_movement_preds[i-2, 0] + 1j * (
        absolute_movement_preds[i-2, 1]))
      next_abs_x = absolute_movement_preds[i-1, 0]
      next_abs_y = absolute_movement_preds[i-1, 1]
      next_rel_rot_x = np.cos(prev_pred_angle)*next_abs_x + np.sin(
        prev_pred_angle)*next_abs_y
      next_rel_rot_y = -np.sin(prev_pred_angle)*next_abs_x + np.cos(
        prev_pred_angle)*next_abs_y
      pred_abs_angle = np.angle(next_rel_rot_x + 1j * next_rel_rot_y)
      
      abs_angle_error = np.abs(pred_abs_angle - cand_rel_angles)
      angle_error = np.minimum(
        np.abs(abs_angle_error), 2*np.pi - abs_angle_error)
      absolute_movement_angle_penalties = (
        abs_movement_angle_constant*angle_error)
      # angle_density_pen = 1/np.exp(-angle_error**2/(2*(1/1)**2))-1
      # absolute_movement_angle_penalties = (
      #   abs_movement_angle_constant*angle_density_pen)
    
    abs_move_uncertainty_mult = min(4, (
      0.2/abs_move_uncertainty_preds[i-1])**abs_move_uncertainty_exponent)
    absolute_movement_penalties = abs_move_uncertainty_mult*(
      absolute_movement_pos_penalties + cum_absolute_movement_pos_penalties + (
        exclude_half_plain_abs_penalties) + absolute_movement_angle_penalties)
    
    # Time leak penalties
    time_leak_preceeding_penalties = np.zeros_like(prev_penalties)
    time_leak_succeeding_penalties = np.zeros_like(prev_penalties)
    # if fn_time_leak.shape[0] == 0:
    #   import pdb; pdb.set_trace()
    #   x=1
    if i == 1 and fn_time_leak.reliable_preceding.values[0]:
      time_leak_preceeding_penalties = time_leak_edge(
        fn_time_leak.delay_preceding, time_leak_delay_cutoff,
        time_leak_time_decay_constant, time_leak_exact_constant,
        time_leak_nearby_constant, time_leak_distance_pen_limit_constant,
        time_leak_dissimilarity_decay, fn_time_leak.preceding_dissimilarity,
        fn_time_leak[['preceding_x', 'preceding_y']],
        floor_waypoints[next_candidates[:, 0]])
      time_leak_preceeding_penalties = np.minimum(
        time_leak_preceeding_penalties, time_leak_max_penalty)
    if i == (num_pred_waypoints-1) and (
        fn_time_leak.reliable_succeeding.values[0]):
      time_leak_succeeding_penalties = time_leak_edge(
        fn_time_leak.delay_succeeding, time_leak_delay_cutoff,
        time_leak_time_decay_constant, time_leak_exact_constant,
        time_leak_nearby_constant, time_leak_distance_pen_limit_constant,
        time_leak_dissimilarity_decay, fn_time_leak.succeeding_dissimilarity,
        fn_time_leak[['succeeding_x', 'succeeding_y']],
        floor_waypoints[next_candidates[:, -1]])
      time_leak_succeeding_penalties = np.minimum(
        time_leak_succeeding_penalties, time_leak_max_penalty)
    combined_leak_penalties = time_leak_preceeding_penalties + (
      time_leak_succeeding_penalties)
      
    # Relative movement penalties
    if i == 1:
      relative_movement_penalties = np.zeros_like(prev_penalties)
    else:
      # Original penalty: absolute error based
      relative_movement_pos_penalties = rel_movement_pos_constant*(np.abs(
        cand_rel_rot_x-relative_movement_preds[i-2, 0]) + (
            np.abs(cand_rel_rot_y-relative_movement_preds[i-2, 1])))
              
      # Updated penalty: angle error based
      pred_rel_angle = np.angle(relative_movement_preds[i-2, 0] + 1j * (
        relative_movement_preds[i-2, 1]))
      rel_angle_error = np.abs(pred_rel_angle - cand_rel_angles)
      angle_error = np.minimum(
        np.abs(rel_angle_error), 2*np.pi - rel_angle_error)
      relative_movement_angle_penalties = (
        rel_movement_angle_constant*angle_error)
      # angle_density_pen = 1/np.exp(-angle_error**2/(2*(1/1)**2))-1
      # relative_movement_angle_penalties = (
      #   rel_movement_angle_constant*angle_density_pen)
      # import pdb; pdb.set_trace()
      
      relative_movement_penalties = (abs_move_uncertainty_mult**0)*(
        relative_movement_pos_penalties + relative_movement_angle_penalties)
      
    step_off_grid_penalties = off_grid_penalties[next_candidates[:, -1]]
    step_addit_grid_density_penalties = addit_grid_density_penalties[
      next_candidates[:, -1]]
      
    candidate_penalties_raw = np.stack([
      prev_penalties,
      wifi_penalties,
      self_repeat_penalties,
      distance_penalties,
      relative_movement_penalties,
      absolute_movement_penalties,
      time_leak_preceeding_penalties,
      time_leak_succeeding_penalties,
      beam_2_no_cheat_penalties,
      wall_penalties,
      wifi_dir_penalties,
      step_off_grid_penalties,
      step_addit_grid_density_penalties,
      magnetometer_abs_pos_penalties,
      ], -1)
    candidate_penalties = candidate_penalties_raw.sum(-1)
    
    # if np.any(np.isnan(candidate_penalties)):
    #   import pdb; pdb.set_trace()
    
    best_candidate_ids = np.argsort(candidate_penalties)[:beam_1_width]
    
    # # Cheat logic - used to verify what's left on the table by non exhaustive
    # # search
    # if all_targets_on_waypoints:
    #   target_id = np.where(np.all(next_candidates == np.expand_dims(
    #     actual_waypoint_ids[:i+1], 0), 1))[0]
    #   if target_id.size == 0:
    #     # import pdb; pdb.set_trace()
    #     x=1
    #   elif not target_id[0] in best_candidate_ids:
    #     best_candidate_ids = np.append(best_candidate_ids, target_id[0])
    
    # Always consider the nearest trajectory in order to obtain debugging stats
    if target_id_nearest.size == 0:
      if valid_mode:
        import pdb; pdb.set_trace()
        x=1
    elif not target_id_nearest[0] in best_candidate_ids:
      # if fn == '5dce884b5516ad00065f03e3':
      #   import pdb; pdb.set_trace()
      #   x=1
      # import pdb; pdb.set_trace()
      # np.where(np.argsort(candidate_penalties) == target_id_nearest[0])
      best_candidate_ids = np.append(best_candidate_ids, target_id_nearest[0])
      
      # Make sure the nearest trajectory never rises back to the top after it
      # dropped out of the beam to avoid cheating during validation
      candidate_penalties[best_candidate_ids[-1]] += 1e6
    
    candidates = next_candidates[best_candidate_ids]
    candidate_penalties = candidate_penalties[best_candidate_ids]
    all_penalties.append((
      candidates,
      wifi_penalties[best_candidate_ids],
      distance_penalties[best_candidate_ids],
      relative_movement_penalties[best_candidate_ids],
      absolute_movement_penalties[best_candidate_ids],
      combined_leak_penalties[best_candidate_ids],
      wall_penalties[best_candidate_ids],
      wifi_dir_penalties[best_candidate_ids],
      step_off_grid_penalties[best_candidate_ids],
      step_addit_grid_density_penalties[best_candidate_ids],
      magnetometer_abs_pos_penalties[best_candidate_ids],
      ))
    
  nearest_pos = floor_waypoints[nearest_waypoint_ids]
  match_rows = np.where(np.all(
    candidates == np.expand_dims(nearest_waypoint_ids, 0), axis=1))[0]
  nearest_rank = match_rows[0] if match_rows.size else np.nan
    
  # if fn == '5dce884b5516ad00065f03e3':
  #   import pdb; pdb.set_trace()
  #   x=1
    
  waypoint_id_preds = candidates[0]
  aligned = floor_waypoints[waypoint_id_preds]
  pred_penalties = get_pred_penalties(all_penalties, candidates[0])
  nearest_penalties = get_pred_penalties(
    all_penalties, candidates[nearest_rank]) if valid_mode else None
  
  # Compute additional optimization statistics
  distance_second = np.sqrt(((
    aligned - floor_waypoints[candidates[1]])**2).sum(1)).sum()
  penalty_gap_second = candidate_penalties[1] - candidate_penalties[0]
  optim_stats = {
    'mean_dist_uncertainty': distance_uncertainty_preds.mean(),
    'mean_abs_move_uncertainty': abs_move_uncertainty_preds.mean(),
    'distance_second': distance_second,
    'penalty_gap_second': penalty_gap_second,
    }
  
  # import pdb; pdb.set_trace()
  return (before_optim_preds, aligned, waypoint_id_preds, pred_penalties,
          nearest_pos, nearest_waypoint_ids, nearest_rank, nearest_penalties,
          target_on_waypoints, time_wifi_min_dist, optim_stats)

def get_optimized_predictions(
    config, valid_mode, site, floor, wifi_floor_preds, floor_waypoints_train,
    this_floor_waypoint_rel_pos_distances, this_floor_waypoint_wifi_distances,
    this_floor_waypoint_wifi_distances_order,
    this_floor_sensor_segment_stats, time_leaks, fn_ids, distance_preds,
    relative_movement_preds, absolute_movement_preds,
    sensor_preds_uncertainties, source_preds, original_preds, source_actual,
    leaderboard_types, ignore_private_test, debug_fn,
    drop_mislabeled_fn_list_valid, w, walls_folder, walls_mode,
    unbias_distance_predictions, verbose):
  optimized_predictions = []
  optimized_predictions_test = []
  numeric_floor = utils.TEST_FLOOR_MAPPING[floor]
  if wifi_floor_preds is None:
    return optimized_predictions, optimized_predictions_test
  
  wifi_pred_vals = wifi_floor_preds.values
  wifi_positions = wifi_pred_vals[:, :2]
  wifi_distances = wifi_pred_vals[:, 2:]
  
  (floor_waypoints, is_train_floor_waypoint, floor_waypoint_types,
   floor_train_waypoint_counts, floor_nearest_known_distances,
   floor_addit_grid_densities) = floor_waypoints_train
  
  # Extract the wifi column mappings for all fns
  wifi_fn_col_ids = {}
  for i, c in enumerate(wifi_floor_preds.columns[2:]):
    parts = c.split('_')
    assert len(parts) == 2
    fn = parts[0]
    time = int(parts[1])
    if fn in wifi_fn_col_ids:
      wifi_fn_col_ids[fn].append((i, time))
    else:
      wifi_fn_col_ids[fn] = [(i, time)]
      
  off_grid_penalties_raw = config['off_grid_waypoint_penalty'] * (
    ~is_train_floor_waypoint)
  num_waypoints = this_floor_waypoint_rel_pos_distances.shape[1]
  act_waypoint_min_distances = ((this_floor_waypoint_rel_pos_distances[
    2]+1e6*np.eye(num_waypoints))[:, is_train_floor_waypoint]).min(1)
  off_grid_penalties = np.maximum(0, off_grid_penalties_raw*(1-(
    act_waypoint_min_distances/config['off_grid_no_penalty_distance'])))
  addit_grid_density_penalties = np.minimum(2, np.maximum(
    0, floor_addit_grid_densities-1))*config['addit_grid_density_penalty']
      
  compute_fns = []
  for fn in wifi_fn_col_ids:
    if (debug_fn is None or fn == debug_fn) and (
        valid_mode or (not ignore_private_test) or (
          leaderboard_types[fn] == 'public')) and (
            not fn in drop_mislabeled_fn_list_valid):
      compute_fns.append(fn)

  walls_count = {}
  if len(compute_fns) and walls_mode:
    site = distance_preds.site.values[distance_preds.fn == compute_fns[0]][0]
    floor = distance_preds.floor.values[distance_preds.fn == compute_fns[0]][0]
    walls_path = walls_folder / (site + '_' + str(floor) + '.pickle')
    if walls_path.is_file():
      with open(walls_path, 'rb') as f:
        walls_count = pickle.load(f)
  orig_walls_count = len(walls_count)

  for fn in wifi_fn_col_ids:
    waypoint_predictions = []
    if fn in compute_fns:
      if psutil.virtual_memory().percent > 95.0:
        raise ValueError("About to run out of memory - ABORT")
      fn_distance_preds = distance_preds[distance_preds.fn == fn]
      fn_relative_movement_preds = relative_movement_preds[
        relative_movement_preds.fn == fn]
      fn_absolute_movement_preds = absolute_movement_preds[
        absolute_movement_preds.fn == fn]
      fn_absolute_movement_preds_vals = fn_absolute_movement_preds[
        ['x_pred', 'y_pred']].values
      fn_sensor_preds_uncertainties = sensor_preds_uncertainties[
        sensor_preds_uncertainties.fn == fn]
      fn_time_leak = time_leaks[time_leaks.fn == fn]
      # fn_sensor_stats = this_floor_sensor_segment_stats[
      #   this_floor_sensor_segment_stats.fn == fn]
      
      wifi_indexes, wifi_times = zip(*wifi_fn_col_ids[fn])
      wifi_indexes = np.array(wifi_indexes)
      wifi_times = np.array(wifi_times)
      wifi_fn_distances = wifi_distances[:, wifi_indexes]
      
      (sensor_fn_distances, sensor_times, sensor_positions,
       this_floor_waypoint_sensor_distances_order) = (
        None, None, None, None)
      
      source_rows = np.where(source_preds.fn == fn)[0]
      waypoint_times = source_preds.waypoint_time[source_rows].values
      actual_waypoints = source_actual[source_rows]
      (before_optim_preds, waypoint_preds, waypoint_id_preds, penalties,
       nearest_pos, nearest_ids, nearest_rank, nearest_penalties,
       target_on_waypoints, min_wifi_dist,
       optim_stats) = align_predictions(
         config, wifi_positions, wifi_times, wifi_fn_distances,
         sensor_positions, sensor_times, sensor_fn_distances, waypoint_times,
         floor_waypoints, off_grid_penalties, addit_grid_density_penalties,
         this_floor_waypoint_rel_pos_distances,
         this_floor_waypoint_wifi_distances,
         this_floor_waypoint_wifi_distances_order,
         this_floor_waypoint_sensor_distances_order, fn_distance_preds,
         fn_relative_movement_preds, fn_absolute_movement_preds,
         fn_sensor_preds_uncertainties, fn_time_leak, actual_waypoints,
         valid_mode, w, walls_count, unbias_distance_predictions)
      all_targets_waypoints = np.all(target_on_waypoints)
      
      if not valid_mode:
        nearest_penalties = np.zeros_like(penalties)*np.nan
      
      num_fn_waypoints = waypoint_times.size
      for i in range(num_fn_waypoints):
        actual = source_actual[source_rows[i]]
        waypoint_pred = waypoint_preds[i]
        train_waypoint_pred = is_train_floor_waypoint[waypoint_id_preds[i]]
        waypoint_type_pred = floor_waypoint_types[waypoint_id_preds[i]]
        waypoint_type_nearest = floor_waypoint_types[nearest_ids[i]]
        waypoint_train_count_pred = floor_train_waypoint_counts[
          waypoint_id_preds[i]]
        waypoint_train_count_nearest = floor_train_waypoint_counts[
          nearest_ids[i]]
        pred_nearest_known_distance = floor_nearest_known_distances[
          waypoint_id_preds[i]]
        before_opt_pred = before_optim_preds[i]
        original_error = np.sqrt(
          ((original_preds[source_rows[i]]-actual)**2).sum())
        after_optim_error = np.sqrt(((waypoint_pred-actual)**2).sum())
        before_optim_error = np.sqrt(((before_opt_pred-actual)**2).sum())
        rel_move_sensor_pred_x = None if i == 0 else (
          fn_absolute_movement_preds_vals[i-1, 0])
        rel_move_sensor_pred_y = None if i == 0 else (
          fn_absolute_movement_preds_vals[i-1, 1])
        segment_dist_pred = None if i == 0 else (
          fn_distance_preds.prediction.values[i-1])
        
        waypoint_predictions.append({
          'site': site,
          'fn': fn,
          'waypoint_time': waypoint_times[i],
          'trajectory_time': waypoint_times[i] - waypoint_times[0],
          'floor': floor,
          'numeric_floor': numeric_floor,
          'trajectory_id': i,
          'num_waypoints': waypoint_times.size,
          'all_targets_on_waypoints': all_targets_waypoints,
          'actual_on_waypoint': target_on_waypoints[i],
          'nearest_trajectory_rank': nearest_rank,
          'nearest_waypoint_x': nearest_pos[i, 0],
          'nearest_waypoint_y': nearest_pos[i, 1],
          'selected_total_penalty': penalties[i].sum(),
          'nearest_total_penalty': nearest_penalties[i].sum(),
          'wifi_penalty': penalties[i, 0],
          'distance_penalty': penalties[i, 1],
          'relative_movement_penalty': penalties[i, 2],
          'absolute_movement_penalty': penalties[i, 3],
          'time_leak_penalty': penalties[i, 4],
          'wall_penalty': penalties[i, 5],
          'wifi_dir_penalty': penalties[i, 6],
          'off_grid_penalty': penalties[i, 7],
          'off_grid_density_penalty': penalties[i, 8],
          'abs_pos_magn_penalty': penalties[i, 9],
          'nearest_wifi_penalty': nearest_penalties[i, 0],
          'nearest_distance_penalty': nearest_penalties[i, 1],
          'nearest_relative_movement_penalty': nearest_penalties[i, 2],
          'nearest_absolute_movement_penalty': nearest_penalties[i, 3],
          'nearest_time_leak_penalty': nearest_penalties[i, 4],
          'nearest_wall_penalty': nearest_penalties[i, 5],
          'nearest_wifi_dir_penalty': nearest_penalties[i, 6],
          'nearest_off_grid_penalty': nearest_penalties[i, 7],
          'nearest_off_grid_density_penalty': nearest_penalties[i, 8],
          'nearest_abs_pos_magn_penalty': nearest_penalties[i, 9],
          'min_wifi_distance': min_wifi_dist[i],
          'train_waypoint_pred': train_waypoint_pred,
          'waypoint_type_pred': waypoint_type_pred,
          'waypoint_type_nearest': waypoint_type_nearest,
          'waypoint_train_count_pred': waypoint_train_count_pred,
          'waypoint_train_count_nearest': waypoint_train_count_nearest,
          'mean_dist_uncertainty': optim_stats['mean_dist_uncertainty'],
          'mean_abs_move_uncertainty': optim_stats[
            'mean_abs_move_uncertainty'],
          'segment_dist_pred': segment_dist_pred,
          'distance_second': optim_stats['distance_second'],
          'penalty_gap_second': optim_stats['penalty_gap_second'],
          'pred_smallest_known_distance': pred_nearest_known_distance,
          'rel_move_sensor_pred_x': rel_move_sensor_pred_x,
          'rel_move_sensor_pred_y': rel_move_sensor_pred_y,
          'x_before_optim_pred': before_opt_pred[0],
          'y_before_optim_pred': before_opt_pred[1],
          'x_pred': waypoint_pred[0],
          'y_pred': waypoint_pred[1],
          'x_actual': actual[0],
          'y_actual': actual[1],
          'original_error': original_error,
          'before_optim_error': before_optim_error,
          'after_optim_error': after_optim_error,
          })
      fn_predictions = pd.DataFrame(waypoint_predictions)
      
      
      if not valid_mode:
        del_cols = []
        for k in list(waypoint_predictions[0].keys()):
          if 'actual' in k or 'error' in k or 'target' in k or 'nearest' in k:
            del_cols.append(k)
       
        fn_predictions.drop(columns=del_cols, inplace=True)
        fn_predictions['leaderboard_type'] = leaderboard_types[fn]
      
      optimized_predictions.append(fn_predictions)
      if valid_mode:
        orig_error = fn_predictions.original_error.mean()
        after_error = fn_predictions.after_optim_error.mean()
        if verbose:
          # import pdb; pdb.set_trace()
          print(f"{fn_ids[fn]}  {orig_error:.2f}  ({after_error:.2f})  {fn} ({num_fn_waypoints})")
      else:
        test_waypoint_predictions = []
        for i in range(waypoint_times.size):
          waypoint_pred = waypoint_preds[i]
          test_waypoint_predictions.append({
            'site_path_timestamp': site + '_' + fn + '_' + str(
              waypoint_times[i]).zfill(13),
            'floor': numeric_floor,
            'x': waypoint_pred[0],
            'y': waypoint_pred[1],
            })
        fn_predictions_test = pd.DataFrame(test_waypoint_predictions)
        optimized_predictions_test.append(fn_predictions_test)
        
        if verbose:
          print(f"{fn_ids[fn]}  {fn} ({num_fn_waypoints})")
    elif not valid_mode:
      source_rows = np.where(source_preds.fn == fn)[0]
      waypoint_times = source_preds.waypoint_time[source_rows].values
      test_waypoint_predictions = []
      for i in range(waypoint_times.size):
        test_waypoint_predictions.append({
          'site_path_timestamp': site + '_' + fn + '_' + str(
            waypoint_times[i]).zfill(13),
          'floor': 0,
          'x': 0,
          'y': 0,
          })
      fn_predictions_test = pd.DataFrame(test_waypoint_predictions)
      optimized_predictions_test.append(fn_predictions_test)
    
  if len(walls_count) != orig_walls_count:
    with open(walls_path, 'wb') as handle:
      pickle.dump(walls_count, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return optimized_predictions, optimized_predictions_test
    
def combined_predictions_all_floors(
    mode, config, use_multiprocessing, distance_preds, relative_movement_preds,
    absolute_movement_preds, sensor_preds_uncertainties, source_preds,
    original_preds, source_actual, sensor_segment_stats, fn_ids, sites, floors,
    time_leaks, wifi_preds_flat, unique_floor_waypoints,
    floor_waypoint_rel_pos_distances, floor_waypoint_wifi_distances,
    floor_waypoint_wifi_distances_order, leaderboard_types,
    ignore_private_test, debug_fn, drop_mislabeled_fn_list_valid, w,
    walls_folder, unbias_distance_predictions, verbose=True):
  valid_mode = mode == 'valid'
  walls_mode = config['wall_penalty_constant'] > 0
  if use_multiprocessing:
    # import pdb; pdb.set_trace()
    # wifi_preds[k]
    # unique_floor_waypoints[k]
    # floor_waypoint_rel_pos_distances[k]
    # floor_waypoint_wifi_distances[k]
    # floor_waypoint_wifi_distances_order[k]
    
    if walls_mode:
      w_copies = [copy.copy(w) for _ in sites]
    else:
      w_copies = [None for _ in sites]

    with mp.Pool(processes=mp.cpu_count()-1) as pool:
      results = [pool.apply_async(
        get_optimized_predictions, args=(
          config, valid_mode, s, f, wpf, u, fwr, fw, fwo, sss, t, fid,
          distance_preds, relative_movement_preds, absolute_movement_preds,
          sensor_preds_uncertainties, source_preds, original_preds,
          source_actual, leaderboard_types, ignore_private_test, debug_fn,
          drop_mislabeled_fn_list_valid, wc, walls_folder, walls_mode,
          unbias_distance_predictions, verbose)) for (
            s, f, wpf, u, fwr, fw, fwo, sss, t, fid, wc) in zip(
              sites, floors, wifi_preds_flat, unique_floor_waypoints,
              floor_waypoint_rel_pos_distances, floor_waypoint_wifi_distances,
              floor_waypoint_wifi_distances_order, sensor_segment_stats,
              time_leaks, fn_ids, w_copies)]
      all_outputs = [p.get() for p in results]
  else:
    all_outputs = []
    for i, (site, floor) in enumerate(zip(sites, floors)):
      all_outputs.append(
        get_optimized_predictions(
          config, valid_mode, site, floor, wifi_preds_flat[i],
          unique_floor_waypoints[i], floor_waypoint_rel_pos_distances[i],
          floor_waypoint_wifi_distances[i], floor_waypoint_wifi_distances_order[i],
          sensor_segment_stats[i], time_leaks[i], fn_ids[i], distance_preds,
          relative_movement_preds, absolute_movement_preds,
          sensor_preds_uncertainties, source_preds, original_preds,
          source_actual, leaderboard_types, ignore_private_test, debug_fn,
          drop_mislabeled_fn_list_valid, copy.copy(w), walls_folder,
          walls_mode, unbias_distance_predictions, verbose))
  
  optimized_predictions = [it for sub in all_outputs for it in sub[0]]
  if not optimized_predictions:
    print("Debug fn not from the correct valid/test mode?")
  optimized_test_predictions = [it for sub in all_outputs for it in sub[1]]
  optimized_predictions = pd.concat(optimized_predictions)
  optimized_test_predictions = None if mode == 'valid' else pd.concat(
    optimized_test_predictions)
  
  return optimized_predictions, optimized_test_predictions
