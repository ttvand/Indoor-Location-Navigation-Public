import numpy as np
import pandas as pd
from pathlib import Path
import pickle

import utils

def run():
  print("Summarizing all waypoint locations")
  only_process_test_sites = True
  write_all_wifi_data = False
  
  data_folder = utils.get_data_folder()
  summary_path = data_folder / "file_summary.csv"
  combined_waypoints_path = data_folder / "train_waypoints_timed.csv"
  if combined_waypoints_path.is_file():
    return
  # combined_train_wifi_times_path = data_folder / "train_wifi_times.csv"
  # combined_test_wifi_times_path = data_folder / "test_wifi_times.csv"
  stratified_holdout_path = data_folder / 'holdout_ids.csv'
  combined_all_wifi_folder = data_folder / 'train'
  df = pd.read_csv(summary_path)
  holdout = pd.read_csv(stratified_holdout_path)
  
  # Loop over all file paths and compare the parquet and pickle files one by one
  all_waypoints = []
  all_train_wifi_times = []
  all_test_wifi_times = []
  all_wifi_data = []
  for i in range(df.shape[0]):
    # if i < 26900:
    #   continue
    
    print(f"Trajectory {i+1} of {df.shape[0]}")
    if (not only_process_test_sites or df.test_site[i]) and df.num_wifi[i] > 0:
      pickle_path = data_folder / (
          str(Path(df.ext_path[i]).with_suffix("")) + "_reshaped.pickle")
      with open(pickle_path, "rb") as f:
        trajectory = pickle.load(f)
        
      if df['mode'][i] != 'test':
        waypoints = trajectory['waypoint']
        num_waypoints = waypoints.shape[0]
        
        # Add meta columns
        for c in ['site_id', 'mode', 'fn', 'text_level']:
          waypoints[c] = df[c][i]
        
        # Add whether it is a train or validation trajectory
        waypoints['mode'] = holdout['mode'][holdout.fn == df.fn[i]].values[0]
        
        # Add the waypoint type
        waypoint_types = np.repeat('middle', num_waypoints)
        waypoint_types[0] = 'first'
        waypoint_types[num_waypoints-1] = 'last'
        waypoints['type'] = waypoint_types
        waypoints['id'] = np.arange(num_waypoints)
        waypoints['num_waypoints'] = num_waypoints
      
      # Add the most recent wifi times that are closest to the waypoint
      # timestamps
      wifi_t1_times = np.unique(trajectory['wifi'].t1_wifi)
      assert np.all(np.diff(wifi_t1_times) > 0)
      wifi_last_t2_times = trajectory['wifi'].groupby(
        't1_wifi')['t2_wifi'].aggregate("max").values
      num_wifi_obs = trajectory['wifi'].groupby(
        't1_wifi')['t1_wifi'].aggregate("count").values
      try:
        assert wifi_t1_times.size == wifi_last_t2_times.size
        assert np.sum(np.diff(wifi_last_t2_times) < -1) <= 1
        assert np.all(wifi_last_t2_times < wifi_t1_times) or (
          df['mode'][i] == 'test')
      except:
        import pdb; pdb.set_trace()
        x=1
        
      if df['mode'][i] != 'test':
        waypoint_wifi_times = np.zeros(num_waypoints, dtype=np.int64)
        for j in range(num_waypoints):
          wifi_id = max(0, (wifi_last_t2_times <= waypoints.time[j]).sum() - 1)
          waypoint_wifi_times[j] = wifi_last_t2_times[wifi_id]
        waypoints['last_wifi_t2_time'] = waypoint_wifi_times
        waypoints['trajectory_wifi_time'] = waypoint_wifi_times - (
          wifi_t1_times[0])
        waypoint_times = waypoints.time.values
        waypoints['trajectory_waypoint_time'] = waypoint_times - (
          waypoint_times[0])
        waypoints['first_waypoint_time'] = waypoint_times[0]
        
        # Reorder the columns
        cols = waypoints.columns.tolist()
        reordered_cols = cols[:1] + cols[4:] + cols[1:4]
        waypoints = waypoints[reordered_cols]
        
        all_waypoints.append(waypoints)
      
      if write_all_wifi_data:
        wifi_data = trajectory['wifi'].copy()
        for c in ['site_id', 'mode', 'fn', 'level']:
          wifi_data[c] = df[c][i]
        cols = wifi_data.columns.tolist()
        reordered_cols = cols[6:] + cols[:6]
        wifi_data = wifi_data[reordered_cols]
        if 'wifi_waypoints' in trajectory:
          wifi_wp = trajectory['wifi_waypoints']
          wifi_wp.sort_values(
            ["t1_wifi", "t2_wifi"], ascending=[True, False], inplace=True)
          wifi_wp_map = wifi_wp.groupby(['t1_wifi']).first().reset_index()[[
            't1_wifi', 'waypoint_interp_x', 'waypoint_interp_y']]
          
          wifi_data = wifi_data.merge(wifi_wp_map, on='t1_wifi')
        else:
          wifi_data['waypoint_interp_x'] = np.nan
          wifi_data['waypoint_interp_y'] = np.nan
        all_wifi_data.append(wifi_data)
      
      wifi_times = pd.DataFrame({
        'site_id': df['site_id'][i],
        'mode': df['mode'][i],
        'fn': df['fn'][i],
        'level': df['level'][i],
        'wifi_t1_times': wifi_t1_times,
        'wifi_last_t2_times': wifi_last_t2_times,
        'trajectory_index': np.arange(wifi_last_t2_times.size),
        'num_wifi_obs': num_wifi_obs,
        })
      
      if df['mode'][i] == 'test':
        wifi_times['first_last_t2_time'] = wifi_last_t2_times[0]
        all_test_wifi_times.append(wifi_times)
      else:
        wifi_times['first_waypoint_time'] = waypoint_times[0]
        all_train_wifi_times.append(wifi_times)
  
  # Write the combined waypoints to disk
  combined_waypoints = pd.concat(all_waypoints)
  combined_waypoints.sort_values(
    ["site_id", "first_waypoint_time", "time"], inplace=True)
  combined_waypoints.to_csv(combined_waypoints_path, index=False)
  
  # # Write the combined wifi times to disk
  # combined_train_wifi_times = pd.concat(all_train_wifi_times)
  # combined_train_wifi_times.sort_values(
  #   ["site_id", "first_waypoint_time", "wifi_t1_times"], inplace=True)
  # combined_train_wifi_times.to_csv(combined_train_wifi_times_path, index=False)
  
  # combined_test_wifi_times = pd.concat(all_test_wifi_times)
  # combined_test_wifi_times.sort_values(
  #   ["site_id", "first_last_t2_time", "wifi_t1_times"], inplace=True)
  # combined_test_wifi_times.to_csv(combined_test_wifi_times_path, index=False)
  
  # Write the raw wifi data to disk
  if write_all_wifi_data:
    test_floors = utils.get_test_floors(data_folder)
    combined_all_wifi = pd.concat(all_wifi_data)
    combined_all_wifi.sort_values(
      ["site_id", "fn", "mode"], inplace=True)
    all_levels = [l if m != 'test' else test_floors[fn] for (l, m, fn) in zip(
      combined_all_wifi.level, combined_all_wifi['mode'], combined_all_wifi.fn)]
    combined_all_wifi['level'] = np.array(all_levels)
    sites = np.sort(np.unique(combined_all_wifi.site_id.values))
    for site_id, site in enumerate(sites):
      print(f"Site {site_id+1} of {len(sites)}")
      combined_all_wifi_site = combined_all_wifi[
        combined_all_wifi.site_id.values == site]
      
      # Map the levels from a reference submission for the test data
      levels = np.sort(np.unique(combined_all_wifi_site.level.values))
      for l in levels:
        combined_all_wifi_floor = combined_all_wifi_site[
          combined_all_wifi_site.level.values == l]
        combined_all_wifi_floor.sort_values(["mode", "t1_wifi"], inplace=True)
        text_level = df.text_level[
          df.fn == combined_all_wifi_floor.fn.values[-1]].values[0]
        
        combined_all_wifi_path = combined_all_wifi_folder / site / text_level / (
          'all_wifi.csv')
        combined_all_wifi_floor.to_csv(combined_all_wifi_path, index=False)