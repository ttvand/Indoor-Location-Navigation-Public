import numpy as np
import pandas as pd
import pickle

import non_parametric_wifi_utils
import utils

analysis_fn = [
  '5dd5216b50e04e0006f56476',
  '5dcd020823759900063d5370',
  '5d119438ffe23f0008604e38',
  '5dd4dab3d48f840006f144eb',
  '5da6f56bab07440006432685',
  '5da4332e1261860006629e59',
  '5dc7b45217ffdd0006f1235f',
  ][-1]
models_group_name = 'non_parametric_wifi'

config = {
  'min_train_points': 10, # Ignore bssid with few observations
  'min_train_fns': 1, # Ignore bssid with few trajectories
  'delay_decay_penalty_exp_base': 0.62, # Base for bssid weight decay as a f of delay to compute the shared bssid fraction
  'inv_fn_count_penalty_exp': 0.1, # Exponent to give more weight to rare bssids to compute the shared bssid fraction
  'non_shared_penalty_start': 1.0, # Threshold below which the shared wifi fraction gets penalized in the distance calculation
  'non_shared_penalty_exponent': 2.2, # Exponent to penalize the non shared wifi fraction
  'non_shared_penalty_constant': 75, # Multiplicative constant to penalize the non shared wifi fraction
  'delay_decay_exp_base': 0.925, # Base for shared bssid weight decay as a f of delay
  'inv_fn_count_distance_exp': 0.1, # Exponent to give more weight to rare bssids to compute the weighted mean distance
  'unique_model_frequencies': False, # Discard bssid's with changing freqs
  'time_range_max_strength': 0, # Group wifi observations before and after each observation and retain the max strength
  'limit_train_near_waypoints': not True, # Similar to "snap to grid" - You likely want to set this to False eventually to get more granular predictions
  }

data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'

summary = pd.read_csv(summary_path)
target_fn = summary.iloc[np.where(summary.fn == analysis_fn)[0][0]]

# Load the relevant raw data
trajectory_path = data_folder / 'train' / target_fn.site_id / (
  target_fn.text_level) / (analysis_fn + '_reshaped.pickle')
with open(trajectory_path, 'rb') as f:
  trajectory = pickle.load(f)
  
# Load the relevant model
model_folder = data_folder.parent / 'Models' / models_group_name
site_model_folder = model_folder / target_fn.site_id
model_path = site_model_folder / (target_fn.text_level + '.pickle')
with open(model_path, 'rb') as f:
  model = pickle.load(f)
  
wifi_groups = dict(tuple(trajectory['wifi'].groupby('t1_wifi')))
t1_wifi_times = np.array(list(wifi_groups.keys()))

wifi_pos_preds = {k: non_parametric_wifi_utils.triangulate(
  wifi_groups[k], model, False, config, None, return_distances=True) for k in (
    wifi_groups)}
    
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
  closest_train_wifi.sort_values(["rssid_wifi"], ascending=False, inplace=True)
  closest_train_wifi.index = np.arange(closest_train_wifi.shape[0])
  
  return closest_train_wifi

def get_shared_bssid(inference, other_wifi):
  shared_bssid = list(set(inference.bssid_wifi).intersection(set(
    other_wifi.bssid_wifi)))
  shared_values = []
  for b in shared_bssid:
    infer_row = np.where(inference.bssid_wifi == b)[0][0]
    other_row = np.where(other_wifi.bssid_wifi == b)[0][0]
    infer = inference.rssid_wifi.values[infer_row]
    train = other_wifi.rssid_wifi.values[other_row]
    delay_inference = inference.delay.values[infer_row]
    delay_train = other_wifi.delay.values[other_row]
    
    shared_values.append({
      'bssid': b,
      'inference': infer,
      'train': train,
      'difference': np.abs(infer-train),
      'delay_train': delay_train,
      'delay_inference': delay_inference,
      })
  shared_bssid = pd.DataFrame(shared_values)
  
  return shared_bssid

def missing_bssid_obs(first, second):
  shared_bssid = list(
    set(first.bssid_wifi).intersection(set(second.bssid_wifi)))
  keep_rows = np.array([i for i, b in enumerate(first.bssid_wifi) if (
    not b in shared_bssid)])
  
  return first.iloc[keep_rows]
    
# Look up the closest point in the model for all waypoints and study their wifi
# strengths
waypoints = trajectory['waypoint'].loc[:, ['x_waypoint', 'y_waypoint']].values
num_waypoints = waypoints.shape[0]
waypoint_positions = np.array(list(model['waypoints'].values()))
inference_closest_historical = []
errors = []
for i in range(num_waypoints):
  waypoint = waypoints[i]
  squared_distances = (
    (np.expand_dims(waypoint, 0)-waypoint_positions)**2).sum(1)
  min_distance_id = np.argmin(squared_distances)
  label_error = np.sqrt(squared_distances[min_distance_id])
  closest_label = get_closest_wifi(model, min_distance_id, waypoint_positions)
  
  closest_inference_id = np.argmin(np.abs(t1_wifi_times - trajectory[
    'waypoint'].time[i]))
  inference = wifi_groups[t1_wifi_times[closest_inference_id]]
  wifi_last_times = inference.groupby(
    't1_wifi')['t2_wifi'].transform("max").values
  inference['delay'] = wifi_last_times-inference['t2_wifi']
  pred_min_distance_id = np.argmin(
    wifi_pos_preds[t1_wifi_times[closest_inference_id]])
  closest_pred = get_closest_wifi(
    model, pred_min_distance_id, waypoint_positions)
  pred_error = np.sqrt(squared_distances[pred_min_distance_id])
  errors.append(pred_error)
  
  inference_not_label = missing_bssid_obs(inference, closest_label)
  label_not_inference = missing_bssid_obs(closest_label, inference)
  
  # Extract the shared bssid's
  shared_bssid_label = get_shared_bssid(inference, closest_label)
  shared_bssid_pred = get_shared_bssid(inference, closest_pred)
  
  inference_closest_historical.append({
    'inference': inference,
    'closest_label': closest_label,
    'closest_pred': closest_pred,
    'inference_not_label': inference_not_label,
    'label_not_inference': label_not_inference,
    'shared_bssid_label': shared_bssid_label,
    'shared_bssid_pred': shared_bssid_pred,
    'pred_error': pred_error,
    'label_error': label_error,
    })
  
print(f"Mean error: {np.array(errors).mean()}")