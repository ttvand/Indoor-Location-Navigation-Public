from collections import ChainMap
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

import utils

mode = ['valid', 'test'][0]
models_group_name = 'sensor_signature'
overwrite_models = True
recompute_grouped_data = not True
debug_fn = [None, '5dc7b45217ffdd0006f1235f'][0]
# '5dc7b46f1cda3700060314b8', '5dcf8361878f3300066c6dd9',
config = {
  'signature_cols': [
    # 'x_ahrs', 'y_ahrs', 'z_ahrs', 'x_magn', 'y_magn',
    'z_magn'],
  'signature_functions': [('mean', np.mean), ('std', np.std)],
  'dist_min_std_quantile': 0.1,
  'limit_train_near_waypoints': True,
  'near_waypoint_time_cutoff_s': 1,
  'train_sensor_frequency_s': 1,
  'signature_half_window_s': 5,
  'min_sensor_frequency': 40, # Default: 50 Hz - drop if large gaps
  }
only_public_test_preds = not True
store_valid_magn_z = True

data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
stratified_holdout_path = data_folder / 'holdout_ids.csv'
leaderboard_types_path = data_folder / 'leaderboard_type.csv'
preds_folder = data_folder.parent / 'Models' / models_group_name / (
    'predictions')
model_folder = data_folder.parent / 'Models' / models_group_name
sensor_folder = data_folder / 'sensor_data'
save_ext = ' - limit near waypoint' if config[
  'limit_train_near_waypoints'] else ''
if not 'df' in locals() or not 'holdout_df' in locals() or (
    not 'test_waypoint_times' in locals()) or (
      not 'test_type_mapping' in locals()):
  df = pd.read_csv(summary_path)
  holdout_df = pd.read_csv(stratified_holdout_path)
  test_waypoint_times = utils.get_test_waypoint_times(data_folder)
  test_floors = utils.get_test_floors(data_folder)
  leaderboard_types = pd.read_csv(leaderboard_types_path)
  test_type_mapping = {fn: t for (fn, t) in zip(
    leaderboard_types.fn, leaderboard_types['type'])}

def multiple_floors_train_predict(
    config, df, models_group_name, mode, holdout_df, test_floors,
    overwrite_models, test_type_mapping, only_public_test_preds,
    test_waypoint_times, debug_fn, verbose=True):
  data_folder = utils.get_data_folder()
  model_folder = data_folder.parent / 'Models' / models_group_name
  site_floors = df.iloc[df.test_site.values].groupby(
    ['site_id', 'text_level']).size().reset_index()
  sites = site_floors.site_id.values
  floors = site_floors.text_level.values
  
  if debug_fn is not None:
    target_row = np.where(df.fn == debug_fn)[0][0]
    sites = [df.site_id.values[target_row]]
    floors = [df.text_level.values[target_row]]
  
  all_distances = []
  all_magnetometer = []
  for floor_id, (analysis_site, floor) in enumerate(zip(sites, floors)):
    if verbose:
      print(f"Processing floor {floor_id+1} of {site_floors.shape[0]}")
    distances, magnetometer_pos = floor_train_predict(
      config, analysis_site, floor, floor_id, data_folder, model_folder, df,
      mode, holdout_df, test_floors, recompute_grouped_data, overwrite_models,
      test_type_mapping, only_public_test_preds, test_waypoint_times, debug_fn)
    all_distances.append(distances)
    all_magnetometer.append(magnetometer_pos)
      
  return all_distances, all_magnetometer

def floor_train_predict(
    config, analysis_site, floor, floor_id, data_folder, model_folder, df,
    mode, holdout_df, test_floors, recompute_grouped_data, overwrite_models,
    test_type_mapping, only_public_test_preds, test_waypoint_times, debug_fn):
  combined_full_sensor_sims = {}
  floor_magnetometer = []
  
  site_model_folder = model_folder / analysis_site
  Path(site_model_folder).mkdir(parents=True, exist_ok=True)

  floor_key = (analysis_site, floor)
  site_df = df[df.site_id == analysis_site]
  if mode != 'test':
    site_df = site_df[site_df['mode'] != 'test']
    valid_paths = holdout_df.ext_path[(holdout_df['mode'] == 'valid') & (
        holdout_df.site_id == analysis_site)].tolist()
    with pd.option_context('mode.chained_assignment', None):
      site_df['mode'] = site_df['ext_path'].apply(
          lambda x: 'valid' if (x in valid_paths) else 'train')
  else:
    test_df = site_df[(site_df['mode'] == 'test')]
    
  floor_df = site_df[site_df.text_level == floor]
  numeric_floor = utils.TEST_FLOOR_MAPPING[floor]
  if mode == 'test':
    target_floors = np.array(
      [test_floors[fn] for fn in test_df['fn'].values])
    correct_test_floor = target_floors == numeric_floor
    if not np.any(correct_test_floor) and only_public_test_preds:
      return combined_full_sensor_sims, floor_magnetometer
    test_df_floor = test_df[correct_test_floor]

  # Load the combined floor train data
  if mode == 'test':
    with pd.option_context("mode.chained_assignment", None):
      floor_df.loc[:, 'mode'] = 'all_train'
    train = utils.load_site_floor(
      floor_df, recompute_grouped_data, test_floor=floor)
    if not np.any(correct_test_floor):
      # Create the all train file, but don't continue since there is nothing to
      # predict
      return combined_full_sensor_sims, floor_magnetometer
    valid = utils.load_site_floor(
      test_df_floor, recompute_grouped_data, test_floor=floor)
  else:
    train = utils.load_site_floor(
      floor_df[floor_df['mode'] == 'train'], recompute_grouped_data)
    valid = utils.load_site_floor(
      floor_df[floor_df['mode'] == 'valid'], recompute_grouped_data)

  # Train the sensor models
  model_type_prefix = 'test-' if mode == 'test' else ''
  model_path = site_model_folder / (model_type_prefix + floor + '.pickle')

  if model_path.exists() and not overwrite_models:
    with open(model_path, 'rb') as f:
      model = pickle.load(f)
  else:
    model = fit_model(train, config, floor_magnetometer)

    with open(model_path, 'wb') as handle:
      pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # Generate predictions with the sensor models
  make_predict_efforts = [v['file_meta']['mode'] != 'test' or (
    test_type_mapping[v['file_meta'].fn] == 'public') or (
      not only_public_test_preds) for v in valid]
  full_sensor_sims = []
  for j, v in enumerate(valid):
    debug_mode = False
    if debug_fn is not None and v['file_meta'].fn == debug_fn:
      debug_mode = True
    elif debug_fn is not None:
      continue
    full_sensor_sim = sensor_sim_trajectory(
      v, make_predict_efforts[j], model, config, floor_magnetometer,
      debug_mode)
    if isinstance(full_sensor_sim, dict):
      if len(full_sensor_sims):
        for k in full_sensor_sim:
          full_sensor_sim[k].drop(['x', 'y'], axis=1, inplace=True)
      full_sensor_sims.append(full_sensor_sim)
  assert len(full_sensor_sims) > 0
  all_full_sensor_sims = {}
  for k in full_sensor_sims[0]:
    all_full_sensor_sims[k] = pd.concat([
      s[k] for s in full_sensor_sims], axis=1)
    all_full_sensor_sims[k] = all_full_sensor_sims[k].astype(np.float16)
  combined_full_sensor_sims[floor_key] = all_full_sensor_sims
      
  floor_magnetometer = pd.concat(floor_magnetometer)
  floor_magnetometer.insert(0, 'floor', floor)
  floor_magnetometer.insert(0, 'site', analysis_site)
  floor_magnetometer.index = np.arange(floor_magnetometer.shape[0])
  
  return combined_full_sensor_sims, floor_magnetometer


def get_sensor_pos_windows(t, config):
  signature_cols = config['signature_cols']
  signature_functions = config['signature_functions']
  near_waypoint_time_cutoff = config['near_waypoint_time_cutoff_s']*1000
  signature_half_window = config['signature_half_window_s']*1000
  train_sensor_frequency = config['train_sensor_frequency_s']*1000
  min_sensor_frequency = config['min_sensor_frequency']
  
  shared_time_raw = t['shared_time']
  shared_time = t['shared_time'].time.values
  waypoint_times = t['waypoint_times']
  waypoint = t.get('waypoint', None)
  fn = t['file_meta'].fn
  window_time = max(shared_time[0]+signature_half_window, waypoint_times[0])
  end_time = min(shared_time[-1]-signature_half_window, waypoint_times[-1])
  sensor_windows = []
  while window_time < end_time:
    start_sensor_time = window_time-signature_half_window
    end_sensor_time = min(window_time+signature_half_window, shared_time[-1])
    center_sensor_time = (start_sensor_time+end_sensor_time)//2
    start_sensor_row = np.argmax(shared_time >= start_sensor_time)
    end_sensor_row = np.argmax(shared_time > end_sensor_time)
    num_sensor_obs = end_sensor_row-start_sensor_row
    window_frequency = num_sensor_obs/(end_sensor_time-start_sensor_time)*1000
    
    interp_id = (window_time >= waypoint_times).sum()-1
    assert interp_id >= 0 and interp_id < (waypoint_times.size-1)
    if waypoint is None:
      waypoint_interp_x = None
      waypoint_interp_y = None
    else:
      frac = (window_time - waypoint_times[interp_id]) / (
        waypoint_times[interp_id+1] - waypoint_times[interp_id])
      waypoint_interp_x = waypoint.x_waypoint[interp_id] + frac * (
          waypoint.x_waypoint[interp_id+1] - waypoint.x_waypoint[interp_id])
      waypoint_interp_y = waypoint.y_waypoint[interp_id] + frac * (
          waypoint.y_waypoint[interp_id+1] - waypoint.y_waypoint[interp_id])
    waypoint_min_time_diff = min(window_time-waypoint_times[interp_id],
                                 waypoint_times[interp_id+1]-window_time)
    near_waypoint_t = waypoint_min_time_diff <= near_waypoint_time_cutoff
    
    if window_frequency >= min_sensor_frequency:
      window_summary = {
        'fn': fn,
        'start_sensor_time': start_sensor_time,
        'center_sensor_time': center_sensor_time,
        'end_sensor_time': end_sensor_time,
        'start_sensor_row': start_sensor_row,
        'end_sensor_row': end_sensor_row,
        'num_sensor_obs': num_sensor_obs,
        'window_frequency': window_frequency,
        'waypoint_interp_x': waypoint_interp_x,
        'waypoint_interp_y': waypoint_interp_y,
        'waypoint_min_time_diff': waypoint_min_time_diff,
        'near_waypoint_t': near_waypoint_t,
        }
      
      for c in signature_cols:
        for f_name, f in signature_functions:
          k = c + '_' + f_name
          window_summary[k] = f(
            shared_time_raw[c].values[start_sensor_row:end_sensor_row])
      
      sensor_windows.append(window_summary)
    window_time += train_sensor_frequency
  
  if len(sensor_windows) == 0:
    return None
  else:
    return pd.DataFrame(sensor_windows)


def fit_model(trajectories, config, floor_magnetometer):
  limit_train_near_waypoints = config['limit_train_near_waypoints']
  dist_min_std_quantile = config['dist_min_std_quantile']
  signature_cols = config['signature_cols']
  
  for t in trajectories:
    t['sensor_pos_windows'] = get_sensor_pos_windows(t, config)
    
    if t['sensor_pos_windows'] is not None:
      train_floor_magnetometer = t['sensor_pos_windows'][[
        'center_sensor_time', 'near_waypoint_t', 'waypoint_interp_x',
        'waypoint_interp_y', 'z_magn_mean', 'z_magn_std']].copy()
      train_floor_magnetometer['fn'] = t['file_meta'].fn
      train_floor_magnetometer['mode'] = 'train'
      floor_magnetometer.append(train_floor_magnetometer)
    
  if limit_train_near_waypoints:
    all_signatures = pd.concat([
        t['sensor_pos_windows'].loc[t['sensor_pos_windows'].near_waypoint_t, :]
        for t in trajectories if t['sensor_pos_windows'] is not None
    ])
  else:
    all_signatures = pd.concat([t['sensor_pos_windows'] for t in trajectories])
  all_signatures.sort_values(['waypoint_interp_x', 'waypoint_interp_y'],
                             inplace=True)
  all_signatures.index = np.arange(all_signatures.shape[0])
  
  dist_min_std = {c: np.quantile(
    all_signatures[c+'_std'].values, dist_min_std_quantile) for c in (
      signature_cols)}

  model = {
      'all_signatures': all_signatures,
      'dist_min_std': dist_min_std,
  }
  
  return model


def signature_distance(
    model_signature, signature, sensor_col, col_dist_min_std):
  mean_col = sensor_col + '_mean'
  std_col = sensor_col + '_std'
  model_mean = model_signature[mean_col].values
  model_std = model_signature[std_col].values
  signature_mean = signature[mean_col]
  signature_std = signature[std_col]
  
  distance_divisor = np.maximum(col_dist_min_std, np.sqrt(
    model_std**2+signature_std**2))
  distances = np.abs(model_mean-signature_mean)/distance_divisor
  
  return distances


def sensor_sim_trajectory(
    t, make_effort, model, config, floor_magnetometer, debug_mode):
  full_pos_dist = []
  if make_effort:
    signature_cols = config['signature_cols']
    model_signature = model['all_signatures']
    dist_min_std = model['dist_min_std']
    
    sensor_pos_windows = get_sensor_pos_windows(t, config)
    fn = t['file_meta'].fn
    full_pos_dist = {c: {
      'x': model_signature.waypoint_interp_x.values,
      'y': model_signature.waypoint_interp_y.values,
      } for c in signature_cols}
    
    for i in range(sensor_pos_windows.shape[0]):
      k = fn + '_' + str(sensor_pos_windows.center_sensor_time.values[i])
      for c in signature_cols:
        full_pos_dist[c][k] = signature_distance(
          model_signature, sensor_pos_windows.iloc[i], c, dist_min_std[c])
    
    val_floor_magnetometer = sensor_pos_windows[[
      'center_sensor_time', 'near_waypoint_t', 'waypoint_interp_x',
      'waypoint_interp_y', 'z_magn_mean', 'z_magn_std']].copy()
    val_floor_magnetometer['fn'] = t['file_meta'].fn
    val_floor_magnetometer['mode'] = 'valid'
    floor_magnetometer.append(val_floor_magnetometer)
      
    for k in full_pos_dist:
      full_pos_dist[k] = pd.DataFrame(full_pos_dist[k])
      
  if debug_mode:
    import pdb; pdb.set_trace()
    x=1
      
  return full_pos_dist

all_distances, all_magnetometer = multiple_floors_train_predict(
    config, df, models_group_name, mode, holdout_df, test_floors,
    overwrite_models, test_type_mapping, only_public_test_preds,
    test_waypoint_times, debug_fn)
if debug_fn is None:
  all_distances = dict(ChainMap(*[o for o in all_distances if o]))
  
  record_time = str(datetime.datetime.now())[:19]
  Path(preds_folder).mkdir(parents=True, exist_ok=True)
  file_ext = models_group_name + ' - ' + mode + save_ext + (
    ' - full distances - ') + record_time + '.pickle'
  full_predictions_path = preds_folder / file_ext
  with open(full_predictions_path, 'wb') as handle:
    pickle.dump(all_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
  if mode == 'valid' and store_valid_magn_z:
    combined_magnetometer = pd.concat(all_magnetometer)
    combined_magnetometer.sort_values([
      'site', 'floor', 'fn', 'center_sensor_time'], inplace=True)
    magn_ext = models_group_name + save_ext + (
      ' - magnetometer - ') + record_time + '.csv'
    combined_magnetometer_path = preds_folder / magn_ext
    import pdb; pdb.set_trace()
    combined_magnetometer.to_csv(combined_magnetometer_path, index=False)