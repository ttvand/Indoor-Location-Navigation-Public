import numpy as np
import pandas as pd
import pickle

import utils

def run():
  print("Inferring device ids")
  only_process_test_sites = True
  signature_dist_threshold = 0.5
  dist_scaler = np.array([
    2.4831865e-03, 1.8569984e-03, 1.5326408e-03,
    3.5197838e+01, 4.1837849e+01, 3.4933647e+01
    ], dtype=np.float32)
  
  sig_cols = [
    'x2_gyro_uncali', 'y2_gyro_uncali', 'z2_gyro_uncali',
    'x2_magn_uncali', 'y2_magn_uncali', 'z2_magn_uncali',
    ]
  data_folder = utils.get_data_folder()
  sensor_folder = data_folder / 'sensor_data'
  summary_path = data_folder / 'file_summary.csv'
  device_id_path = data_folder / 'device_ids.pickle'
  if device_id_path.is_file():
    return
  save_ext = '' if only_process_test_sites else '_all_sites'
  meta_sensor_path = sensor_folder / ('meta' + save_ext + '_no_device.csv')
  df = pd.read_csv(summary_path)
  meta_sensor = pd.read_csv(meta_sensor_path, dtype={'test_type': object})
  if only_process_test_sites:
    df = df[df.test_site]
  
  df.index = np.arange(df.shape[0])
  with pd.option_context('mode.chained_assignment', None):
    df['first_last_wifi_replaced_time'] = df['first_last_wifi_time']
    no_wifi_rows = np.where(df.num_wifi == 0)[0]
    assert np.all(df['mode'].values[no_wifi_rows] == 'train')
    df.loc[no_wifi_rows, 'first_last_wifi_replaced_time'] = (
      df.start_time.values[no_wifi_rows])
  
  df.sort_values(by=['first_last_wifi_replaced_time'], axis=0, inplace=True)
  df.index = np.arange(df.shape[0])
  
  all_sensor = {}
  for m in ['valid', 'test', 'train']:
    print(m)
    with open(sensor_folder / (m + save_ext + '.pickle'), 'rb') as f:
      sensor_data = pickle.load(f)
    all_sensor.update(sensor_data)
  
  fns = df.fn.values
  modes = df['mode'].values
  num_fn = df.shape[0]
  unique_sites = np.sort(np.unique(df.site_id.values))
  
  device_ids = {}
  device_ids_ordered = []
  active_device_signatures = []
  act_sig_recent_ids = []
  next_signature_id = 0
  dev_stats = []
  for i in range(num_fn):
    if (i+1) % 1000 == 0:
      print(i+1)
    fn = fns[i]
    mode = modes[i]
    
    # if fn in ['5daec763aa1d300006faafcd', '5daece4eaa1d300006fab032']:
    #   import pdb; pdb.set_trace()
    #   x=1
    
    first_uncal_vals = all_sensor[fn]['waypoint_segments'][0][sig_cols].values
    last_uncal_vals = all_sensor[fn]['waypoint_segments'][-1][sig_cols].values
    signature_absent = np.isnan(first_uncal_vals[0, 0]) or np.isnan(
      last_uncal_vals[0, 0])
    this_first_signature = first_uncal_vals[0]
    this_last_signature = last_uncal_vals[-1]
    signature_change_this_step = not np.all(
      np.isclose(this_first_signature, this_last_signature))
    plot_time = df.first_last_wifi_time.values[i] if mode == 'test' else (
      df.start_time.values[i])
    meta_rows = np.where(meta_sensor.fn.values == fn)[0]
    mean_robust_sensor_time_diff = np.median(
      meta_sensor.mean_robust_sensor_time_diff.values[meta_rows])
    site_id = np.where(df.site_id.values[i] == unique_sites)[0][0]
    
    if signature_absent:
      device_ids[fn] = (-1, None)
      device_ids_ordered.append((fn, -1, None))
      dev_stats.append({
        'fn': fn, 'device_id': -1, 'site_id': site_id, 'plot_time': plot_time,
        'mean_robust_sensor_time_diff': mean_robust_sensor_time_diff,
        })
      continue
    
    # Compute when the next trajectory can use the same device
    if mode == 'test':
      corrected_start_time = df.first_last_wifi_time.values[i] + 5000
      this_min_next_available_time = df.duration.values[i] + (
        df.first_last_wifi_time.values[i]) - 5000
    else:
      this_min_next_available_time = df.end_time.values[i]
      corrected_start_time = df.start_time.values[i]
    
    found_signature = False
    for j in act_sig_recent_ids:
      (signature, signature_id, min_available_time, prev_sig_mode, prev_fn,
       prev_row, prev_drift) = active_device_signatures[j]
      signature_dist = (np.abs(this_first_signature - signature)/(
        dist_scaler)).sum()
      num_shared_nz = (np.isclose(signature, this_first_signature) & (
        signature != 0)).sum()
      # if signature_dist > 0:
      #   print(signature_dist, num_shared_nz)
      same_signature = signature_dist <= signature_dist_threshold or (
        num_shared_nz > 1)
      
      if same_signature:
        if corrected_start_time < min_available_time:
          print(i, corrected_start_time, min_available_time, signature_dist,
                num_shared_nz)
          print("This should not happen - signature time inconsistency")
        
        # if signature_change_this_step:
        #   import pdb; pdb.set_trace()
        
        device_ids[fn] = (signature_id, signature_change_this_step)
        device_ids_ordered.append((fn, signature_id, signature_change_this_step))
        dev_stats.append({
          'fn': fn, 'device_id': signature_id, 'site_id': site_id,
          'plot_time': plot_time,
          'mean_robust_sensor_time_diff': mean_robust_sensor_time_diff,
        })
        found_signature = True
        active_device_signatures[j] = (
          this_last_signature, signature_id, this_min_next_available_time, mode,
          fn, i, signature_change_this_step)
        act_sig_recent_ids.remove(j)
        act_sig_recent_ids = [j] + act_sig_recent_ids
        break
      
    if not found_signature:
      signature_id = next_signature_id
      # if signature_id == 52:
      #   import pdb; pdb.set_trace()
      #   x=1
      device_ids[fn] = (signature_id, signature_change_this_step)
      device_ids_ordered.append((fn, signature_id, signature_change_this_step))
      dev_stats.append({
          'fn': fn, 'device_id': signature_id, 'site_id': site_id,
          'plot_time': plot_time,
          'mean_robust_sensor_time_diff': mean_robust_sensor_time_diff,
        })
      active_device_signatures.append((
        this_last_signature, signature_id, this_min_next_available_time, mode,
        fn, i, signature_change_this_step))
      act_sig_recent_ids = [next_signature_id] + act_sig_recent_ids
      next_signature_id += 1
  
  combined_signatures = pd.DataFrame(
    np.stack([s[0] for s in active_device_signatures]))
  
  # Stitch device ids back together using time, mean time between sensor
  # observations and the site id.
  # Also split the -1 device ids based on mean time between sensor observations
  dev_stats_df = pd.DataFrame(dev_stats)
  dev_stats_df.loc[(dev_stats_df.device_id.values == -1) & (
    dev_stats_df.mean_robust_sensor_time_diff < 20), 'device_id'] = -2
  predecessors = {
    -2: [],
    -1: [],
    }
  
  stats_device_ids = dev_stats_df.device_id.values
  site_ids = dev_stats_df.site_id.values
  plot_times = dev_stats_df.plot_time.values
  rtds = dev_stats_df.mean_robust_sensor_time_diff.values
  for i in range(dev_stats_df.device_id.values.max()+1):
    first_row = np.where(stats_device_ids == i)[0][0]
    first_rtd = rtds[first_row]
    this_site_id = site_ids[first_row]
    
    pred_candidates = []
    for c in predecessors:
      last_chain_device = c if not len(predecessors[c]) else (
        predecessors[c][-1])
      pred_last_row = np.where(stats_device_ids == last_chain_device)[0][-1]
      pred_last_rtd = rtds[pred_last_row]
      pred_site_id = site_ids[pred_last_row]
      time_gap = plot_times[first_row] - plot_times[pred_last_row]
      print(i, last_chain_device, time_gap)
      
      if time_gap > 0 and time_gap <= 86400000 and (
          this_site_id == pred_site_id) and np.abs(
            first_rtd - pred_last_rtd) < 0.02:
        pred_candidates.append(c)
      
    if len(pred_candidates):
      assert len(pred_candidates) == 1
      predecessors[pred_candidates[0]].append(i)
    else:
      predecessors[i] = []
  
  merged_device_ids = {}
  for k_id, k in enumerate(list(predecessors.keys())):
    merged_device_ids[k] = k_id
    for v in predecessors[k]:
      merged_device_ids[v] = k_id
  
  combined_device_ids = {}
  for fn in device_ids:
    dev_id, drift = device_ids[fn]
    rtd_fn = rtds[np.where(dev_stats_df.fn.values == fn)[0][0]]
    if dev_id == -1 and rtd_fn < 20:
      dev_id = -2
    combined_device_ids[fn] = (dev_id, drift, merged_device_ids[dev_id])
  
  #np.array([v[2] for k, v in combined_device_ids.items()])
  
  with open(device_id_path, 'wb') as handle:
    pickle.dump(combined_device_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)