import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle

import utils

def similarity_wifi_time_edge(first, second, max_delay_s=10):
  if first is None or second is None:
    return -1, -1
  
  last_first = first[first.t1_wifi == first.t1_wifi.max()]
  first_second = second[second.t1_wifi == second.t1_wifi.min()]
  
  # Only consider non delayed observations
  delays_first = (last_first.t2_wifi.max() - last_first.t2_wifi).values
  delays_second = (first_second.t2_wifi.max() - first_second.t2_wifi).values
  
  first_comp = last_first[delays_first/1000 <= max_delay_s]
  second_comp = first_second[delays_second/1000 <= max_delay_s]
  
  shared_ids = set(first_comp.bssid_wifi).intersection(set(
    second_comp.bssid_wifi))
  
  shared_fraction = 2*len(shared_ids)/(
    first_comp.shape[0]+second_comp.shape[0])
  
  shared_distances = np.array([
    first_comp.rssid_wifi[first_comp.bssid_wifi == b].values[0] - (
      second_comp.rssid_wifi[second_comp.bssid_wifi == b].values[0]) for b in (
        shared_ids)])
  if shared_distances.size:
    mean_abs_shared_distance = np.abs(shared_distances).mean()
  else:
    mean_abs_shared_distance = 20
  
  return shared_fraction, mean_abs_shared_distance

def extract_floor_start_end(
    data_folder, site, df, holdout_df, test_preds, device_time_path, mode,
    device_ids, public_private_test_leaks, max_tries=5):
  # Load all wifi data for the site
  all_floors = np.unique(df.text_level.values[(
    df.site_id == site) & (df['mode'] != 'test')])
  train_wifi = {}
  for f in all_floors:
    floor_wifi_path = data_folder / 'train' / site / f / 'all_train.pickle'
    with open(floor_wifi_path, 'rb') as f:
      all_train = pickle.load(f)
    for i in range(len(all_train)):
      if 'wifi' in all_train[i]:
        train_wifi[all_train[i]['file_meta'].fn] = all_train[i]['wifi']
  
  test_wifi = {}
  for f in all_floors:
    floor_wifi_path = data_folder / 'train' / site / f / 'test.pickle'
    try:
      with open(floor_wifi_path, 'rb') as f:
        all_test = pickle.load(f)
    except:
      # No test data for the considered floor
      all_test = []
    for i in range(len(all_test)):
      test_wifi[all_test[i]['file_meta'].fn] = all_test[i]['wifi']
  
  # Update the start and end time for the test data as well as the estimated
  # floor and waypoints
  site_df = df[(df.site_id == site)]
  site_df.index = np.arange(site_df.shape[0])
  site_dtp = device_time_path[device_time_path.site == site]
  site_dtp.index = np.arange(site_dtp.shape[0])
  first_waypoints_site_df = site_df[[
    'first_waypoint_x', 'first_waypoint_y']].values
  last_waypoints_site_df = site_df[[
    'last_waypoint_x', 'last_waypoint_y']].values
  assert (site_df.num_wifi > 0).sum() == (len(train_wifi) + len(test_wifi))
  site_dtp_fn = site_dtp.fn.values.tolist()
  for test_fn in test_wifi:
    test_fn_wifi = test_wifi[test_fn]
    wifi_last_times = test_fn_wifi.groupby(
      't1_wifi')['t2_wifi'].aggregate("max").reset_index()
    offsets = wifi_last_times['t1_wifi'] - wifi_last_times['t2_wifi']
    start_time = -(offsets.values.min()-200)
    wifi_last_times['t1_corrected'] = start_time + wifi_last_times['t1_wifi']
    test_fn_rows = np.where(test_preds.fn == test_fn)[0]
    pred_waypoints = test_preds[['x', 'y']].values[test_fn_rows]
    
    if test_fn in public_private_test_leaks:
      leak_type, leak_floor, leak_x, leak_y = public_private_test_leaks[
        test_fn]
      if leak_type == 'start':
        pred_waypoints[0] = [leak_x, leak_y]
        first_waypoints_site_df[
          np.where(site_df.fn.values == test_fn)[0][0]] = pred_waypoints[0]
      else:
        pred_waypoints[-1] = [leak_x, leak_y]
        last_waypoints_site_df[
          np.where(site_df.fn.values == test_fn)[0][0]] = pred_waypoints[-1]
    
    with pd.option_context('mode.chained_assignment', None):
      site_df_row = np.where(site_df.fn.values == test_fn)[0][0]
      first_pred_row = np.where(test_preds.fn == test_fn)[0][0]
      site_df.loc[site_df_row, 'level'] = test_preds.floor[first_pred_row]
      site_df.loc[site_df_row, 'start_time'] += start_time
      site_df.loc[site_df_row, 'end_time'] += start_time
      
      site_df.loc[site_df_row, 'first_waypoint_x'] = pred_waypoints[0, 0]
      site_df.loc[site_df_row, 'first_waypoint_y'] = pred_waypoints[0, 1]
      site_df.loc[site_df_row, 'last_waypoint_x'] = pred_waypoints[-1, 0]
      site_df.loc[site_df_row, 'last_waypoint_y'] = pred_waypoints[-1, 1]
    
  valid_fns = set(holdout_df.fn[holdout_df['mode'] == 'valid'])
  if mode == 'valid':
    infer_fns = list(
      valid_fns.intersection(set(site_df.fn[site_df.num_wifi > 0])))
  else:
    infer_fns = list(test_wifi.keys())
  
  # For all fns: look at possible matching fns at the start and end:
  # - Compatible start/end times
  # - Compatible locations
  # - Compatible wifi signatures
  # - Compatible floors
  results = []
  for fn in infer_fns:
    meta_fn = site_df.iloc[np.where(site_df.fn == fn)[0][0]]
    actual_device_id = device_ids[fn][0]
    actual_device_id_drift = device_ids[fn][1]
    actual_device_id_merged = device_ids[fn][2]
    
    # Obtain the preceding details
    start_time = meta_fn.start_time
    site_dtp_id = site_dtp_fn.index(fn)
    actual_dev_id = site_dtp.device_id.values[site_dtp_id]
    if actual_dev_id >= 0:
      prev_dev_id = None if site_dtp_id == 0 else site_dtp.device_id.values[
        site_dtp_id-1]
      if prev_dev_id is None or (actual_dev_id != prev_dev_id):
        compatible_preceding_ids = np.zeros(0)
      else:
        target_prec_fn = site_dtp_fn[site_dtp_id-1]
        compatible_preceding_ids = np.where(site_df.fn == target_prec_fn)[0]
    else:
      compatible_preceding_ids = np.where(start_time > site_df.end_time)[0]
    this_wifi = train_wifi.get(fn, test_wifi.get(fn, None))
    if compatible_preceding_ids.size:
      considered_end_ids = np.argsort(
        -site_df.end_time.values[compatible_preceding_ids])[:max_tries]
      considered_end_ids = np.append(considered_end_ids, considered_end_ids[0])
      considered_preceding_floors = []
      for i, latest_end_id in enumerate(considered_end_ids):
        preceding_id = compatible_preceding_ids[latest_end_id]
        preceding_floor = site_df.level[preceding_id]
        delay_preceding = start_time - site_df.end_time.values[preceding_id]
        assert delay_preceding > 0
        preceding_fn = site_df['fn'][preceding_id]
        leak_test_preceding = preceding_fn in public_private_test_leaks and (
          public_private_test_leaks[preceding_fn][0] == 'end')
        preceding_device_id = device_ids[preceding_fn][0]
        preceding_device_id_drift = device_ids[preceding_fn][1]
        preceding_device_id_merged = device_ids[preceding_fn][2]
        prec_valid = preceding_fn in valid_fns
        preceding_mode = 'valid' if prec_valid else site_df['mode'][
          preceding_id]
        can_use_preceding = (
          (preceding_mode != 'test') or leak_test_preceding) and not (
            (mode == 'valid') and prec_valid)
        preceding_wifi = train_wifi.get(
          preceding_fn, test_wifi.get(preceding_fn, None))
        preceding_shared_fraction, preceding_mean_dist = (
          similarity_wifi_time_edge(preceding_wifi, this_wifi))
        preceding_pos = np.full(2, np.nan) if not can_use_preceding else (
          last_waypoints_site_df[preceding_id])
        
        # Estimate if the preceding or succeeding position can be used
        preceding_dissimilarity = (
          10*(1-preceding_shared_fraction) + preceding_mean_dist + 15*int(
            preceding_shared_fraction <= 0.1)) + max(
              0, min(4, np.log2(delay_preceding/10000)))
        # preceding_dissimilarity = (
        #   10*(1-preceding_shared_fraction) + preceding_mean_dist + max(
        #     0, min(4, np.log2(delay_preceding/10000))) + int(
        #       preceding_shared_fraction <= 0.1))
        reliable_preceding = can_use_preceding and (
          preceding_dissimilarity < 12) and (
            preceding_device_id == actual_device_id)
        reliable_preceding_floor = preceding_floor if reliable_preceding else (
          None)
        reliable_preceding_device_id = preceding_device_id if (
          reliable_preceding) else None
        
        if reliable_preceding and not reliable_preceding_floor in (
            considered_preceding_floors):
          break
        considered_preceding_floors.append(preceding_floor)
      preceding_match_offset_fn = i % (considered_end_ids.size - 1)
    else:
      can_use_preceding = False
      delay_preceding = -1
      preceding_fn = None
      preceding_device_id = None
      preceding_device_id_drift = None
      preceding_device_id_merged = None
      preceding_shared_fraction = -1
      preceding_mean_dist = -1
      preceding_mode = 'None'
      preceding_pos = np.full(2, np.nan)
      preceding_dissimilarity = None
      reliable_preceding = False
      reliable_preceding_floor = None
      reliable_preceding_device_id = None
      preceding_match_offset_fn = None
      preceding_floor = None
    
    # if preceding_match_offset_fn is not None and preceding_match_offset_fn > 0:
    #   import pdb; pdb.set_trace()
    #   x=1
    
    
    # Obtain the succeeding details
    end_time = meta_fn.end_time
    if actual_dev_id >= 0:
      next_dev_id = None if site_dtp_id == (site_dtp.shape[0]-1) else (
        site_dtp.device_id.values[site_dtp_id+1])
      if next_dev_id is None or (actual_dev_id != next_dev_id):
        compatible_succeeding_ids = np.zeros(0)
      else:
        target_suc_fn = site_dtp_fn[site_dtp_id+1]
        compatible_succeeding_ids = np.where(site_df.fn == target_suc_fn)[0]
    else:
      compatible_succeeding_ids = np.where(end_time < site_df.start_time)[0]
    if compatible_succeeding_ids.size:
      considered_start_ids = np.argsort(
        site_df.start_time.values[compatible_succeeding_ids])[:max_tries]
      considered_start_ids = np.append(
        considered_start_ids, considered_start_ids[0])
      considered_succeeding_floors = []
      for i, first_start_id in enumerate(considered_start_ids):      
        succeeding_id = compatible_succeeding_ids[first_start_id]
        succeeding_floor = site_df.level[succeeding_id]
        delay_succeeding = site_df.start_time.values[succeeding_id] - end_time
        assert delay_succeeding > 0
        succeeding_fn = site_df['fn'][succeeding_id]
        leak_test_succeeding = succeeding_fn in public_private_test_leaks and (
          public_private_test_leaks[succeeding_fn][0] == 'start')
        succeeding_device_id = device_ids[succeeding_fn][0]
        succeeding_device_id_drift = device_ids[succeeding_fn][1]
        succeeding_device_id_merged = device_ids[succeeding_fn][2]
        suc_valid = succeeding_fn in valid_fns
        succeeding_mode = 'valid' if suc_valid else site_df['mode'][
          succeeding_id]
        can_use_succeeding = (
          (succeeding_mode != 'test') or leak_test_succeeding) and not (
            (mode == 'valid') and suc_valid)
        succeeding_wifi = train_wifi.get(
          succeeding_fn, test_wifi.get(succeeding_fn, None))
        succeeding_shared_fraction, suc_mean_dist = similarity_wifi_time_edge(
          this_wifi, succeeding_wifi)
        succeeding_pos = np.full(2, np.nan) if not can_use_succeeding else (
          first_waypoints_site_df[succeeding_id])
        
        # Estimate if the preceding or succeeding position can be used
        succeeding_dissimilarity = (
          10*(1-succeeding_shared_fraction) + suc_mean_dist + 15*int(
            succeeding_shared_fraction <= 0.1)) + max(
              0, min(4, np.log2(delay_succeeding/10000)))
        # succeeding_dissimilarity = (
        #   10*(1-succeeding_shared_fraction) + suc_mean_dist + max(
        #     0, min(4, np.log2(delay_succeeding/10000))) + int(
        #       succeeding_shared_fraction <= 0.1))
        reliable_succeeding = can_use_succeeding and (
          succeeding_dissimilarity < 12) and (
            succeeding_device_id == actual_device_id)
        reliable_succeeding_floor = succeeding_floor if (
          reliable_succeeding) else None
        reliable_succeeding_device_id = succeeding_device_id if (
          reliable_succeeding) else None
        
        if reliable_succeeding and not reliable_succeeding_floor in (
            considered_succeeding_floors):
          break
        considered_succeeding_floors.append(succeeding_floor)
      succeeding_match_offset_fn = i % (considered_start_ids.size -1)
    else:
      can_use_succeeding = False
      delay_succeeding = -1
      succeeding_fn = None
      succeeding_device_id = None
      succeeding_device_id_drift = None
      succeeding_device_id_merged = None
      succeeding_shared_fraction = -1
      suc_mean_dist = -1
      succeeding_mode = 'None'
      succeeding_pos = np.full(2, np.nan)
      succeeding_dissimilarity = None
      reliable_succeeding = False
      reliable_succeeding_floor = None
      reliable_succeeding_device_id = None
      succeeding_match_offset_fn = None
      succeeding_floor = None
      
    # if succeeding_match_offset_fn is not None and succeeding_match_offset_fn > 0:
    #   import pdb; pdb.set_trace()
    #   x=1
      
    num_waypoints = meta_fn.num_test_waypoints if meta_fn[
      'mode'] == 'test' else meta_fn.num_train_waypoints
    results.append({
      'site': site,
      'fn': fn,
      'num_waypoints': num_waypoints,
      
      'can_use_preceding': can_use_preceding,
      'delay_preceding': delay_preceding,
      'preceding_shared_fraction': preceding_shared_fraction,
      'preceding_mean_dist': preceding_mean_dist,
      'preceding_dissimilarity': preceding_dissimilarity,
      'preceding_mode': preceding_mode,
      'preceding_fn': preceding_fn,
      'preceding_device_id': preceding_device_id,
      'preceding_device_id_drift': preceding_device_id_drift,
      'preceding_device_id_merged': preceding_device_id_merged,
      'preceding_match_offset_fn': preceding_match_offset_fn,
      'preceding_floor': preceding_floor,
      'preceding_x': preceding_pos[0],
      'preceding_y': preceding_pos[1],
      'first_x': meta_fn.first_waypoint_x,
      'first_y': meta_fn.first_waypoint_y,
      
      'can_use_succeeding': can_use_succeeding,
      'delay_succeeding': delay_succeeding,
      'succeeding_shared_fraction': succeeding_shared_fraction,
      'succeeding_mean_dist': suc_mean_dist,
      'succeeding_dissimilarity': succeeding_dissimilarity,
      'succeeding_mode': succeeding_mode,
      'succeeding_fn': succeeding_fn,
      'succeeding_device_id': succeeding_device_id,
      'succeeding_device_id_drift': succeeding_device_id_drift,
      'succeeding_device_id_merged': succeeding_device_id_merged,
      'succeeding_floor': succeeding_floor,
      'succeeding_match_offset_fn': succeeding_match_offset_fn,
      'succeeding_x': succeeding_pos[0],
      'succeeding_y': succeeding_pos[1],
      'last_x': meta_fn.last_waypoint_x,
      'last_y': meta_fn.last_waypoint_y,
      
      'reliable_preceding': reliable_preceding,
      'reliable_succeeding': reliable_succeeding,
      'reliable_preceding_device_id': reliable_preceding_device_id,
      'reliable_succeeding_device_id': reliable_succeeding_device_id,
      'actual_device_id': actual_device_id,
      'actual_device_id_drift': actual_device_id_drift,
      'actual_device_id_merged': actual_device_id_merged,
      'reliable_preceding_floor': reliable_preceding_floor,
      'reliable_succeeding_floor': reliable_succeeding_floor,
      'actual_floor': meta_fn.level,
      })
    
  site_results = pd.DataFrame(results)
  
  return site_results

def run(mode):
  debug_site = [None, 0][0]
  use_multiprocessing = False
  test_preds_source = 'test - 2021-05-15 05:19:44.csv'
  test_override_floors = False
  
  data_folder = utils.get_data_folder()
  test_override_ext = '_floor_override' if (
    mode == 'test' and test_override_floors) else ''
  save_path = data_folder / (
    mode + '_edge_positions_v3' + test_override_ext + '.csv')
  if save_path.is_file():
    return
  
  summary_path = data_folder / 'file_summary.csv'
  test_preds_path = data_folder / 'submissions' / test_preds_source
  stratified_holdout_path = data_folder / 'holdout_ids.csv'
  device_id_path = data_folder / 'device_ids.pickle'
  ordered_device_time_path = data_folder / 'inferred_device_ids.csv'
  with open(device_id_path, 'rb') as f:
    device_ids = pickle.load(f)
  public_private_test_leaks = {
    'ff141af01177f34e9caa7a12': ('start', 3, 203.11885, 97.310814),
    'f973ee415265be4addc457b1': ('start', -1, 20.062187, 99.66188),
    '23b4c8eb4b41d75946285461': ('end', 2, 60.205635, 102.28055),
    '5582270fcaee1f580de9006f': ('end', 0, 97.8957	, 28.9133),
    'b51a662297b90657f0b03b44': ('start', 1, 112.39258	, 233.72379),
    }
  df = pd.read_csv(summary_path)
  holdout_df = pd.read_csv(stratified_holdout_path)
  test_floors = utils.get_test_floors(
    data_folder, debug_test_floor_override=test_override_floors)
  test_preds = pd.read_csv(test_preds_path)
  test_preds = utils.override_test_floor_errors(
    test_preds, debug_test_floor_override=test_override_floors)
  test_preds['fn'] = [
    spt.split('_')[1] for spt in test_preds.site_path_timestamp]
  test_preds['timestamp'] = [
    int(spt.split('_')[2]) for spt in test_preds.site_path_timestamp]
  for test_fn in test_floors:
    assert test_preds.floor[test_preds.fn == test_fn].values[0] == test_floors[
      test_fn]
    
  device_time_path = pd.read_csv(ordered_device_time_path)
  device_time_path['time'] = device_time_path['start_time']
  test_rows = np.where(device_time_path['mode'].values == "test")[0]
  device_time_path.loc[test_rows, 'time'] = device_time_path[
    'first_last_wifi_time'].values[test_rows]
  device_time_path.sort_values(['device_id', 'time'], inplace=True)
  
  sites = df.iloc[df.test_site.values].groupby(
    ['site_id']).size().reset_index()
  if debug_site is not None:
    sites = sites.iloc[debug_site:(debug_site+1)]
  sites = sites.site_id.values
  
  if use_multiprocessing:
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
      results = [pool.apply_async(
        extract_floor_start_end, args=(
          data_folder, s, df, holdout_df, test_preds, device_time_path,
          mode, device_ids, public_private_test_leaks)) for s in sites]
      all_outputs = [p.get() for p in results]
  else:
    all_outputs = []
    for site_id, analysis_site in enumerate(sites):
      print(f"Processing site {site_id+1} of {len(sites)}")
      all_outputs.append(
        extract_floor_start_end(
          data_folder, analysis_site, df, holdout_df, test_preds,
          device_time_path, mode, device_ids, public_private_test_leaks))
      
  # Save the combined results
  combined = pd.concat(all_outputs)
  combined.to_csv(save_path, index=False)