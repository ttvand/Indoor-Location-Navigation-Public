import copy
import numpy as np
import pandas as pd
import pathlib

import utils

def run(mode):
  print("Preparing features for the sensor uncertainty models")
  fn_mode = ['mean', 'first_middle_last'][0]
  num_neighbors = 10
  additional_feature_cols = ['rel_fraction', 'time_offset']
  feature_cols = ['total_dist', 'num_waypoints']
  stem_target_cols = ['mean_rel_dist_error', 'mean_abs_rel_dist_error',
                      'mean_angle_error', 'mean_abs_angle_error']
  
  if fn_mode == 'mean':
    target_cols = copy.copy(stem_target_cols)
  else:
    target_cols = []
    for ext in ['first_', 'middle_', 'last_']:
      for c in stem_target_cols:
        if ext == 'middle_':
          target_cols += [ext + c]
        else:
          target_cols += [ext + c[5:]]
  feature_cols += copy.copy(target_cols)
  
  data_folder = utils.get_data_folder()
  model_folder = data_folder.parent / 'Models' / 'correct_sensor_preds'
  pathlib.Path(model_folder).mkdir(parents=True, exist_ok=True)
  save_ext = '' if fn_mode == 'mean' else ' first_middle_last'
  save_path = model_folder / (mode + save_ext + '.csv')
  
  if save_path.is_file():
    return
  
  data_path = data_folder / 'fn_device_errors.csv'
  data = pd.read_csv(data_path)
  num_add_features = len(additional_feature_cols)
  all_feature_cols = additional_feature_cols + feature_cols
  num_keep_first_data_cols = 7
    
  num_shifts = num_neighbors*2+1
  num_rows = data.shape[0]
  num_features = len(feature_cols)
  all_features = np.zeros((num_shifts, num_rows, num_features+num_add_features))
  padded_feature_vals = np.full((num_rows+num_shifts-1, num_features), np.nan)
  padded_feature_vals[num_neighbors:(-num_neighbors)] = data[
    feature_cols].values
  device_ids = data.device_id.values
  padded_device_ids = np.full((num_rows+num_shifts-1), np.nan)
  padded_device_ids[num_neighbors:(-num_neighbors)] = device_ids
  times = data.plot_time.values
  padded_times = np.full((num_rows+num_shifts-1), np.nan)
  padded_times[num_neighbors:(-num_neighbors)] = times
  modes = data['mode'].values
  can_use_mask = np.concatenate([
    np.zeros(num_neighbors, dtype=bool),
    (modes != 'test') & ((mode == 'test') | (modes != 'valid')),
    np.zeros(num_neighbors, dtype=bool),
    ])
  for shift_id, shift in enumerate(range(-num_neighbors, num_neighbors+1)):
    start_row = shift+num_neighbors
    end_row = shift+num_neighbors+num_rows
    shifted_features = np.copy(padded_feature_vals[start_row:end_row])
    shifted_device_ids = padded_device_ids[start_row:end_row]
    if shift == 0:
      step_can_use_mask = np.ones_like(can_use_mask[start_row:end_row])
    else:
      step_can_use_mask = np.copy(can_use_mask[start_row:end_row])
    shift_mask = step_can_use_mask & (shifted_device_ids == device_ids)
    shifted_features[~shift_mask] = np.nan
    time_offsets = padded_times[start_row:end_row] - times
    sign_log_time_offsets = np.sign(time_offsets) * np.log10(
      np.abs(time_offsets))
    sign_log_time_offsets [~shift_mask] = np.nan
    
    all_features[shift_id, :, 1] = sign_log_time_offsets
    all_features[shift_id, :, num_add_features:] = shifted_features
  
  # Add the weighted mean distance within the window (excluding the centered fn)
  all_features[num_neighbors, :, 2] = np.nan
  dist_surrounding_sum = np.nansum(all_features[:, :, 2], 0, keepdims=True)
  all_features[:, :, 0] = all_features[:, :, 2]/dist_surrounding_sum
  
  # Convert the features and targets to a flat dataframe
  df_cols = {}
  for i in range(num_keep_first_data_cols):
    col = data.columns[i]
    df_cols[col] = data[col].values
  df_cols['device_id'] = data.device_id.values
  df_cols['plot_time'] = data.plot_time.values
  no_middle_segments = data.num_waypoints.values <= 3
  for k in target_cols:
    target_col_id = np.where(np.array(feature_cols) == k)[0][0] + (
      num_add_features)
    target_vals = np.copy(all_features[num_neighbors, :, target_col_id])
    if fn_mode == 'first_middle_last' and k[:6] == 'middle':
      target_vals[no_middle_segments] = np.nan
    df_cols[k+'_target'] = target_vals
  for c_id, c in enumerate(all_feature_cols):
    for shift_id, shift in enumerate(range(-num_neighbors, num_neighbors+1)):
      if shift != 0:
        col_name = c + str(shift)
        df_cols[col_name] = all_features[shift_id, :, c_id]
        
  combined = pd.DataFrame(df_cols)
  combined.to_csv(save_path, index=False)