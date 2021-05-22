import numpy as np
import pickle

import pandas as pd
import utils

def run():
  only_process_test_sites = True
  data_folder = utils.get_data_folder()
  sensor_folder = data_folder / 'sensor_data'
  device_id_path = data_folder / 'device_ids.pickle'
  try:
    with open(device_id_path, 'rb') as f:
      device_ids = pickle.load(f)
  except:
    device_ids = None
  device_ext = '_no_device' if device_ids is None else ''
  save_ext = '' if only_process_test_sites else '_all_sites'
  save_path = sensor_folder / ('meta' + save_ext + device_ext + '.csv')
  if save_path.is_file():
    return
  summary_path = data_folder / 'file_summary.csv'
  df = pd.read_csv(summary_path)
  leaderboard_types_path = data_folder / 'leaderboard_type.csv'
  leaderboard_types = pd.read_csv(leaderboard_types_path)
  test_type_mapping = {fn: t for (fn, t) in zip(
    leaderboard_types.fn, leaderboard_types['type'])}
  
  # Combine all the sub-trajectory meta data
  all_sub_trajectories = []
  for mode in ['test', 'train', 'valid']:
    print(mode)
    load_path = sensor_folder / (mode + save_ext + '.pickle')
    with open(load_path, 'rb') as f:
      combined_mode = pickle.load(f)
      
    for fn in combined_mode:
      t = combined_mode[fn]
      
      site = t['site']
      level = t['floor']
      text_level = df.text_level.values[np.where(
        (df.site_id == site) & (df.level == level))[0][0]]
      num_waypoints = t['num_waypoints']
      waypoint_times = t['waypoint_times']
      sub_durations = np.diff(waypoint_times)
      
      waypoint_segments = t['waypoint_segments']
      waypoint_times = t['waypoint_times']
      relative_movements = t['relative_waypoint_movement_1']
      for i in range(num_waypoints-1):
        segment_time = waypoint_segments[i].time.values
        sensor_time_diff = np.diff(segment_time)
        start_time_offset = segment_time[0] - waypoint_times[i] 
        end_time_offset = segment_time[-1] - waypoint_times[i+1] 
        mean_robust_sensor_time_diff = sensor_time_diff[
          (sensor_time_diff >= 19) & (sensor_time_diff <= 21)].mean()
        
        if mode == 'test':
          distance_covered = None
          test_type = test_type_mapping[fn]
          plot_time = df.first_last_wifi_time.values[
            np.where(df.fn.values == fn)[0][0]]
        else:
          distance_covered = np.sqrt((relative_movements[i]**2).sum())
          test_type = ''
          plot_time = waypoint_times[i]
        
        all_sub_trajectories.append({
          'mode': mode,
          'site': site,
          'level': level,
          'text_level': text_level,
          'fn': fn,
          'device_id': None if device_ids is None else device_ids[fn][0],
          'device_id_merged': None if device_ids is None else (
            device_ids[fn][2]),
          'test_type': test_type,
          
          'plot_time': plot_time,
          'start_time': waypoint_times[i],
          'end_time': waypoint_times[i+1],
          'sub_trajectory_id': i,
          'num_waypoints': num_waypoints,
          'duration': sub_durations[i],
          'num_obs': segment_time.size,
          'start_time_offset': start_time_offset,
          'end_time_offset': end_time_offset,
          'mean_sensor_time_diff': sensor_time_diff.mean(),
          'mean_robust_sensor_time_diff': mean_robust_sensor_time_diff,
          'min_sensor_time_diff': sensor_time_diff.min(),
          'max_sensor_time_diff': sensor_time_diff.max(),
          'distance_covered': distance_covered,
          })
        
  combined = pd.DataFrame(all_sub_trajectories)
  combined.to_csv(save_path, index=False)