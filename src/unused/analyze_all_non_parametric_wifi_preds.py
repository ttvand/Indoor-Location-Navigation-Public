import datetime
import numpy as np
import pandas as pd
import pickle

import multiprocessing as mp

import non_parametric_wifi_utils
import utils

models_group_name = 'non_parametric_wifi'
debug_floor = [None, 12][1]
use_multiprocessing = not True

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
  'limit_train_near_waypoints': True # Similar to "snap to grid" - You likely want to set this to False eventually to get more granular predictions
  }

data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
stratified_holdout_path = data_folder / 'holdout_ids.csv'
holdout_df = pd.read_csv(stratified_holdout_path)
summary = pd.read_csv(summary_path)

all_valid_fn = holdout_df.iloc[
  np.where((holdout_df['mode'] == 'valid') & (holdout_df.test_site))[0]]
all_valid_fn = all_valid_fn.sort_values(
  ["site_id", "text_level"], ascending=True)

def get_closest_wifi(model, min_distance_id, waypoint_positions):
  closest_waypoint = (waypoint_positions[min_distance_id, 0],
                      waypoint_positions[min_distance_id, 1])
  closest_waypoint_wifi = []
  for j, b in enumerate(model['bssid']):
    strength = model['strengths'][min_distance_id, j]
    if strength < 0:
      closest_waypoint_wifi.append({
        't1_wifi': model['t1_waypoints'][closest_waypoint],
        'source_fn': model['waypoint_fns'][closest_waypoint],
        'bssid_wifi': b,
        'rssid_wifi': strength,
        'delay':  model['delays'][min_distance_id, j],
        })
  closest_train_wifi = pd.DataFrame(closest_waypoint_wifi)
  if closest_train_wifi.shape[0] > 1:
    closest_train_wifi.sort_values(
      ["rssid_wifi"], ascending=False, inplace=True)
    closest_train_wifi.index = np.arange(closest_train_wifi.shape[0])
  
  return closest_train_wifi

def get_shared_bssid(inference, other_wifi):
  if other_wifi.shape[0] == 0:
    shared_bssid = []
    non_shared_values = inference.rssid_wifi.values
  else:
    shared_bssid = list(set(inference.bssid_wifi).intersection(set(
      other_wifi.bssid_wifi)))
    
    non_shared_inference_values = inference.rssid_wifi.values[np.array([
      not b in shared_bssid for b in inference.bssid_wifi])]
    non_shared_other_values = other_wifi.rssid_wifi.values[np.array([
      not b in shared_bssid for b in other_wifi.bssid_wifi])]
    non_shared_values = np.concatenate([
      non_shared_inference_values, non_shared_other_values])
    
  non_shared_mean_strength = non_shared_values.mean()
  shared_values = []
  for b in shared_bssid:
    infer = inference[inference.bssid_wifi == b].rssid_wifi.values[0]
    train = other_wifi[other_wifi.bssid_wifi == b].rssid_wifi.values[0]
    shared_values.append({
      'bssid': b,
      'inference': infer,
      'train': train,
      'difference': np.abs(infer-train),
      })
  shared_bssid = pd.DataFrame(shared_values)
  
  return shared_bssid, non_shared_mean_strength

def get_floor_waypoint_predictions(
    analysis_site, floor, floor_id, all_valid_fn):
  floor_waypoint_predictions = []
  print(f"Floor {floor_id+1} of {site_floors.shape[0]}")
  
  # Load the relevant model
  model_folder = data_folder.parent / 'Models' / models_group_name
  site_model_folder = model_folder / analysis_site
  model_path = site_model_folder / (floor + '.pickle')
  with open(model_path, 'rb') as f:
    model = pickle.load(f)
  
  analysis_fns = all_valid_fn.fn[(all_valid_fn.site_id == analysis_site) & (
    all_valid_fn.text_level == floor)].values
  
  for analysis_fn in analysis_fns:
    # Load the relevant raw data
    trajectory_path = data_folder / 'train' / analysis_site / floor / (
      analysis_fn + '_reshaped.pickle')
    with open(trajectory_path, 'rb') as f:
      trajectory = pickle.load(f)
      
    wifi_groups = dict(tuple(trajectory['wifi'].groupby('t1_wifi')))
    t1_wifi_times = np.array(list(wifi_groups.keys()))
    wifi_pos_preds = {k: non_parametric_wifi_utils.triangulate(
      wifi_groups[k], model, config, None, return_distances=True) for k in (
        wifi_groups)}
    
    # Look up the closest point in the model for all waypoints and study their
    # wifi strengths
    waypoints = trajectory['waypoint'].loc[
      :, ['x_waypoint', 'y_waypoint']].values
    num_waypoints = waypoints.shape[0]
    waypoint_positions = np.array(list(model['waypoints'].values()))
    trajectory_waypoint_times = trajectory['waypoint'].time
    for i in range(num_waypoints):
      waypoint = waypoints[i]
      squared_distances = (
        (np.expand_dims(waypoint, 0)-waypoint_positions)**2).sum(1)
      min_distance_id = np.argmin(squared_distances)
      label_error = np.sqrt(squared_distances[min_distance_id])
      closest_label = get_closest_wifi(model, min_distance_id, waypoint_positions)
      
      waypoint_time = trajectory_waypoint_times[i]
      closest_inference_id = np.argmin(np.abs(t1_wifi_times-waypoint_time))
      inference = wifi_groups[t1_wifi_times[closest_inference_id]]
      pred_min_distance_id = np.argmin(
        wifi_pos_preds[t1_wifi_times[closest_inference_id]][1])
      closest_pred = get_closest_wifi(
        model, pred_min_distance_id, waypoint_positions)
      pred_error = np.sqrt(squared_distances[pred_min_distance_id])
      pred_rank = 1+(
        squared_distances < squared_distances[pred_min_distance_id]).sum()
      
      # Extract the shared bssid's
      shared_bssid_label, non_shared_label_mean_strength = get_shared_bssid(
        inference, closest_label)
      shared_bssid_pred, non_shared_pred_mean_strength = get_shared_bssid(
        inference, closest_pred)
      
      shared_label_mean_distance = shared_bssid_label.difference.mean() if (
        shared_bssid_label.shape[0] > 0) else None
      shared_pred_mean_distance = shared_bssid_pred.difference.mean() if (
        shared_bssid_pred.shape[0] > 0) else None
      
      label_fn = utils.get_most_freq_np_str(
        np.concatenate(closest_label.source_fn.values))
      # import pdb; pdb.set_trace()
      floor_waypoint_predictions.append({
        'site': analysis_site,
        'floor': floor,
        'fn': analysis_fn,
        'label_fn': label_fn,
        'waypoint_time': waypoint_time,
        'num_inference': inference.shape[0],
        
        'num_closest_label': closest_label.shape[0],
        'num_shared_label': shared_bssid_label.shape[0],
        'shared_label_mean_distance': shared_label_mean_distance,
        # 'non_shared_label_mean_strength': non_shared_label_mean_strength,
        
        'num_closest_pred': closest_pred.shape[0],
        'num_shared_pred': shared_bssid_pred.shape[0],
        'shared_pred_mean_distance': shared_pred_mean_distance,
        # 'non_shared_pred_mean_strength': non_shared_pred_mean_strength,
        
        'label_error': label_error,
        'pred_error': pred_error,
        'pred_rank': pred_rank,
        'optimal_prediction': pred_error == label_error,
        })
      
  return pd.DataFrame(floor_waypoint_predictions)
    
site_floors = all_valid_fn.groupby(
  ['site_id', 'text_level']).size().reset_index()
if debug_floor is not None:
  site_floors = site_floors.iloc[debug_floor:(debug_floor+1)]
sites = site_floors.site_id.values
floors = site_floors.text_level.values

if use_multiprocessing:
  floor_ids = np.arange(floors.size)
  with mp.Pool(processes=mp.cpu_count()-1) as pool:
    results = [pool.apply_async(
      get_floor_waypoint_predictions, args=(
        s, f, i, all_valid_fn)) for (s, f, i) in zip(
          sites, floors, floor_ids)]
    all_floor_waypoint_predictions = [p.get() for p in results]
else:
  all_floor_waypoint_predictions = []
  for floor_id, (analysis_site, floor) in enumerate(zip(sites, floors)):
    all_floor_waypoint_predictions.append(get_floor_waypoint_predictions(
      analysis_site, floor, floor_id, all_valid_fn))
    
all_predictions = pd.concat(all_floor_waypoint_predictions)
all_predictions.sort_values(["site", "floor"], ascending=True)
preds_folder = data_folder.parent / 'Models' / models_group_name / (
    'predictions')
record_time = str(datetime.datetime.now())[:19]
preds_path = preds_folder / (
  models_group_name + ' - validation pointwise - ' + record_time + '.csv')
print(all_predictions.pred_error.mean())
x=y
all_predictions.to_csv(preds_path, index=False)