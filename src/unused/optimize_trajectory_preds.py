from functools import partial
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.optimize import minimize

import utils

models_group_name = 'non_parametric_wifi'
preds_source = 'non_parametric_wifi - valid - full distances - 2021-03-30 09:02:59.pickle'
initial_pos_top_k = 5
top_distance_pos = 5
near_distance_exp = 0.8
initial_weight_dist_exp = 3
target_move_speed = 5
covered_dist_exp = 1.5
move_speed_tolerance = 1

ref_source = preds_source.replace('- full distances ', '').replace(
  'pickle', 'csv')
data_folder = utils.get_data_folder()
preds_folder = Path(data_folder).parent / 'Models' / models_group_name / (
  'predictions')
preds_path =  preds_folder / preds_source
if not 'preds' in locals() or not 'original_preds' in locals():
  with open(preds_path, 'rb') as f:
    preds = pickle.load(f)
  source_preds = pd.read_csv(preds_folder / ref_source)
  source_actual = source_preds[['x_actual', 'y_actual']].values
  original_preds = source_preds[['x_pred', 'y_pred']].values

def trajectory_loss(wifi_positions, positions, times, distances,
                    ref_wifi_distances, target_move_speed=target_move_speed,
                    move_speed_tolerance=move_speed_tolerance,
                    covered_dist_exp=covered_dist_exp):
  # Compute the penalty for consistency with the wifi signal
  wifi_positions = wifi_positions.reshape((-1, 2))
  num_wifi = wifi_positions.shape[0]
  wifi_error = np.stack([wifi_distance(
    wifi_positions[i], distances[:, i], positions) for i in range(
      num_wifi)]).mean()
  
  # Penalize moving more than you can realistically expect in the time between
  # wifi observations
  distance_covered = np.sqrt(
    ((wifi_positions[:-1] - wifi_positions[1:])**2).sum(1))
  t_elapsed = np.diff(times)/1000
  e_distance_low = (target_move_speed-move_speed_tolerance)*t_elapsed/3.6
  e_distance_high = (target_move_speed+move_speed_tolerance)*t_elapsed/3.6
  covered_distance_error = ((np.maximum(0, e_distance_low-distance_covered) + (
    np.maximum(0, distance_covered-e_distance_high)))**covered_dist_exp).mean()
  
  loss = wifi_error + 3*covered_distance_error
  
  return loss

def wifi_distance(
    pos, wifi_distances, positions, num_top=top_distance_pos,
    near_distance_exp=near_distance_exp):
  pos_distances = np.sqrt(((positions-np.expand_dims(pos, 0))**2).sum(1))
  near_ids = np.argsort(pos_distances)[:initial_pos_top_k]
  near_distances = pos_distances[near_ids]
  near_wifi_distances = wifi_distances[near_ids]
  
  dist_weights = near_distance_exp ** near_distances
  weighted_distance = (near_wifi_distances*dist_weights).sum()/(
    dist_weights.sum())
  
  return weighted_distance

def align_predictions(
    positions, times, fn_distances, waypoint_times, initial_pos_top_k,
    initial_weight_dist_exp):
  num_times = times.size
  assert num_times == fn_distances.shape[1]
  initial_preds = np.zeros((num_times, 2))
  ref_wifi_distances = []
  for i in range(num_times):
    top_ids = np.argsort(fn_distances[:, i])[:initial_pos_top_k]
    top_pos = positions[top_ids]
    top_distances = fn_distances[top_ids, i]
    min_distance = top_distances.min()
    pos_weights = 1/((top_distances+1e-5)/(min_distance+1e-5))**(
      initial_weight_dist_exp)
    weighted_pos = (top_pos*np.expand_dims(pos_weights, 1)).sum(0)/(
      pos_weights.sum())
    initial_preds[i] = weighted_pos
    ref_wifi_distances.append(wifi_distance(
      weighted_pos, fn_distances[:, i], positions))
    
  ref_wifi_distances = np.concatenate([ref_wifi_distances])
    
  loss = partial(
    trajectory_loss, positions=positions, times=times, distances=fn_distances,
    ref_wifi_distances=ref_wifi_distances)
  x0 = initial_preds.flatten().tolist()
  res = minimize(
      loss, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
  time_preds = res.x.reshape((-1, 2))
    
  before_optim_preds = utils.interpolate_preds(
    times, initial_preds, waypoint_times)
  after_optim_preds = utils.interpolate_preds(
    times, time_preds, waypoint_times)
  
  return before_optim_preds, after_optim_preds

optimized_predictions = []
for k in preds:
  floor_preds = preds[k]
  if floor_preds is None:
    continue
  
  site = k[0]
  floor = k[1]
  numeric_floor = utils.TEST_FLOOR_MAPPING[floor]
  
  pred_vals = floor_preds.values
  positions = pred_vals[:, :2]
  distances = pred_vals[:, 2:]
  
  # Extract the column mappings for all fns
  fn_col_ids = {}
  for i, c in enumerate(floor_preds.columns[2:]):
    parts = c.split('_')
    assert len(parts) == 2
    fn = parts[0]
    time = int(parts[1])
    if fn in fn_col_ids:
      fn_col_ids[fn].append((i, time))
    else:
      fn_col_ids[fn] = [(i, time)]
      
  for fn in fn_col_ids:
    indexes, times = zip(*fn_col_ids[fn])
    indexes = np.array(indexes)
    times = np.array(times)
    fn_distances = distances[:, indexes]
    
    source_rows = np.where(source_preds.fn == fn)[0]
    waypoint_times = source_preds.waypoint_time[source_rows].values
    initial_preds, waypoint_preds = align_predictions(
      positions, times, fn_distances, waypoint_times, initial_pos_top_k,
      initial_weight_dist_exp)
    
    waypoint_predictions = []
    for i in range(waypoint_times.size):
      actual = source_actual[source_rows[i]]
      waypoint_pred = waypoint_preds[i]
      before_optim_error = np.sqrt(((initial_preds[i]-actual)**2).sum())
      after_optim_error = np.sqrt(((waypoint_pred-actual)**2).sum())
      original_error = np.sqrt(
        ((original_preds[source_rows[i]]-actual)**2).sum())
      waypoint_predictions.append({
        'site': site,
        'fn': fn,
        'waypoint_time': waypoint_times[i],
        'floor': floor,
        'numeric_floor': numeric_floor,
        'x_pred': waypoint_pred[0],
        'y_pred': waypoint_pred[1],
        'x_actual': actual[0],
        'y_actual': actual[1],
        'original_error': original_error,
        'before_optim_error': before_optim_error,
        'after_optim_error': after_optim_error,
        })
    fn_predictions = pd.DataFrame(waypoint_predictions)
    optimized_predictions.append(fn_predictions)
    orig_error = fn_predictions.original_error.mean()
    before_error = fn_predictions.before_optim_error.mean()
    after_error = fn_predictions.after_optim_error.mean()
    print(f"{orig_error:.2f} ({before_error-orig_error:.2f}) ({after_error-orig_error:.2f})")
    
import pdb; pdb.set_trace()
optimized_predictions = pd.concat(optimized_predictions)
optimized_predictions.sort_values(
  ["site", "floor", "fn", "waypoint_time"], inplace=True)
optimized_error = optimized_predictions.error.mean()
print(f"Optimized validation error: {optimized_error:.2f}")