# import matplotlib.pyplot as plt
import multiprocessing as mp
# import networkx as nx
import numpy as np
import pandas as pd
import utils


chunk_ms = 100000
min_range_margin = 10 # As a multiple of chunk_ms
max_delay_ms = 2000
max_considered_near_distances = 10
debug_floor = [None, 0][1]
use_multiprocessing = False

data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
best_test_predictions_path = data_folder / 'submissions' / 'sub_cost_snap.csv'
if not 'df' in locals() or (not 'test_preds' in locals()) or (
    not 'test_pred_waypoints' in locals()):
  df = pd.read_csv(summary_path)
  test_preds = pd.read_csv(best_test_predictions_path)
  test_pred_fns = np.array([sps.split('_')[1] for sps in (
    test_preds.site_path_timestamp)])
  test_pred_times = np.array([int(sps.split('_')[2]) for sps in (
    test_preds.site_path_timestamp)])
  test_pred_waypoints = test_preds[['x', 'y']].values

def store_floor_bssid_stats(
    analysis_site, floor, floor_id, site_floors, data_folder, test_pred_fns,
    test_pred_times, test_pred_waypoints, chunk_ms,
    max_considered_near_distances, max_delay_ms, min_range_margin):
  print(f"Floor {floor_id+1} of {site_floors.shape[0]}")
  floor_wifi_path = data_folder / 'train' / analysis_site / floor / (
    'all_wifi.csv')
  wifi = pd.read_csv(floor_wifi_path)[[
    'mode', 'fn', 'bssid_wifi', 'rssid_wifi', 't1_wifi', 't2_wifi',
    'waypoint_interp_x', 'waypoint_interp_y']]
  
  # Interpolate the waypoints for the test trajectories
  test_fns = np.unique(wifi.fn[wifi['mode'] == 'test'].values)
  wifi_t1 = wifi['t1_wifi'].values
  for fn in test_fns:
    fn_ids = np.where(wifi.fn == fn)[0]
    num_fn_wifi = fn_ids.size
    fn_t1_times = wifi_t1[fn_ids]
    fn_pred_ids = np.where(test_pred_fns == fn)[0]
    fn_t1_waypoints = test_pred_times[fn_pred_ids]
    fn_waypoints = test_pred_waypoints[fn_pred_ids]
    num_fn_waypoints = fn_pred_ids.size
    interpolated_test = np.zeros((num_fn_wifi, 2))
    interpolated_test += np.expand_dims(fn_waypoints[0], 0)
    
    for i in range(num_fn_waypoints-1):
      fraction = (fn_t1_times - fn_t1_waypoints[i]) / (
        fn_t1_waypoints[i+1] - fn_t1_waypoints[i])
      interpolated_ids = np.where((fraction > 0) & (fraction <= 1))[0]
      
      interpolated_test[interpolated_ids] = np.expand_dims(
        fn_waypoints[i], 0) + np.expand_dims(
          fraction[interpolated_ids], 1) * np.expand_dims(
            fn_waypoints[i+1] - fn_waypoints[i], 0)
    
    if fn_t1_times.max() > fn_t1_waypoints.max():
      interpolated_test[fn_t1_times > fn_t1_waypoints.max()] = (
        np.expand_dims(fn_waypoints[-1], 0))
      
    wifi.loc[fn_ids, ['waypoint_interp_x', 'waypoint_interp_y']] = (
      interpolated_test)
    
  # For each truncated location: compute the first and last chunked t2s
  rounded_x = np.round(wifi.waypoint_interp_x)
  rounded_y = np.round(wifi.waypoint_interp_y)
  chunked_times = np.round(wifi.t2_wifi.values/chunk_ms)
  total_floor_time_range = np.ptp(wifi.t2_wifi.values)
  wifi['rounded_x'] = rounded_x
  wifi['rounded_y'] = rounded_y
  wifi['chunked_time'] = chunked_times.astype(np.int64)
  wifi_last_times = wifi.groupby(
      ['fn', 't1_wifi'])['t2_wifi'].transform("max").reset_index().values[:, 1]
  wifi['delay'] = wifi_last_times-wifi['t2_wifi']
  wifi['t1_count'] = wifi.groupby(['fn', 't1_wifi']).transform(
    "count").reset_index().values[:, 2]
  min_chunked_time = chunked_times.min()
  max_chunked_time = chunked_times.max()
  unique_rounded = np.unique(np.stack([rounded_x, rounded_y], 1), axis=0)
  
  unique_chunked_times = []
  for i in range(unique_rounded.shape[0]):
    x = unique_rounded[i, 0]
    y = unique_rounded[i, 1]
    unique_chunked_times.append(np.sort(np.unique(chunked_times[
      (rounded_x == x) & (rounded_y == y)])))
    
  # Compute near distance ids for thresholds between 1 and 10
  near_distance_ids = {}
  for i in range(unique_rounded.shape[0]):
    x = unique_rounded[i, 0]
    y = unique_rounded[i, 1]
    
    distances = np.sqrt((unique_rounded[:, 0]-x)**2 + (
      unique_rounded[:, 1]-y)**2)
    distances_ids = {}
    for j in range(max_considered_near_distances+1):
      distances_ids[j] = np.where(distances <= j)[0]
      
    near_distance_ids[(x, y)] = distances_ids
    
  # Slicing 1: Compute statistics separately for each bssid
  bssid_grouped = dict(tuple(wifi.groupby("bssid_wifi")))
  bssid_stats = []
  bssid_unique_fns = []
  for b_id, b in enumerate(list(bssid_grouped.keys())):
    wifi_b = bssid_grouped[b]
    # print(b_id, wifi_b.shape[0])
    pos_rss = wifi_b.iloc[wifi_b.delay.values < max_delay_ms].groupby(
      ['rounded_x','rounded_y'])['rssid_wifi'].max().reset_index()
    pos_rss.sort_values(["rssid_wifi"], ascending=False, inplace=True)
    
    min_chunked_b_time = wifi_b.chunked_time.min()
    max_chunked_b_time = wifi_b.chunked_time.max()
    min_range = [max(min_chunked_time, min_chunked_b_time-min_range_margin),
                 min(max_chunked_time, max_chunked_b_time+min_range_margin)]
    max_range = [min_chunked_time-1, max_chunked_time+1]
    
    for i in range(pos_rss.shape[0]):
      rss = pos_rss.rssid_wifi[i]
      x = pos_rss.rounded_x[i]
      y = pos_rss.rounded_y[i]
      max_distance = min(max_considered_near_distances,
                         max(0, int((80 + rss)/8)))
      near_time_ids = near_distance_ids[(x, y)][max_distance]
      near_times = np.concatenate(
        [unique_chunked_times[id_] for id_ in near_time_ids])
      max_reduce_ids = (near_times > min_range[1]) & (
        near_times < (max_range[1] - 1))
      min_increase_ids = (near_times < min_range[0]) & (
        near_times > (max_range[0] + 1))
      
      # Reduce the max range upper limit
      if np.any(max_reduce_ids):
        max_range[1] = (near_times[max_reduce_ids]).max()+1
        
      if np.any(min_increase_ids):
        max_range[0] = (near_times[min_increase_ids]).min()-1
        
    active_range = (np.array(max_range)*chunk_ms).astype(np.int64)
    active_time_range_fraction = min(1, np.diff(active_range)[0]/(
      total_floor_time_range))
    num_non_delayed = pos_rss.shape[0]
    fraction_non_delayed = num_non_delayed/wifi_b.shape[0]
    if num_non_delayed == 0:
      max_strength = None
      mean_non_delayed_strength = None
    else:
      max_strength = pos_rss.rssid_wifi.values.max()
      mean_non_delayed_strength = pos_rss.rssid_wifi.values.mean()
    
    bssid_stats.append({
      'bssid': b,
      'min_active_time': active_range[0],
      'max_active_time': active_range[1],
      'active_time_range_fraction': active_time_range_fraction,
      'num_non_delayed': num_non_delayed,
      'fraction_non_delayed': fraction_non_delayed,
      'max_strength': max_strength,
      'mean_non_delayed_strength': mean_non_delayed_strength,
      'num_fns': np.unique(wifi_b.fn.values).size,
      'mean_t1_count': (wifi_b.t1_count.values).mean(),
      })
    
    bssid_unique_fns.append(np.sort(np.unique(wifi_b.fn.values)))
    
  floor_bssid_stats = pd.DataFrame(bssid_stats)
  import pdb; pdb.set_trace()
  floor_stats_wifi_path = data_folder / 'train' / analysis_site / floor / (
    'bssid_stats.csv')
  floor_bssid_stats.to_csv(floor_stats_wifi_path, index=False)
  
  # Slicing 2: Compute implicit device statistics for each trajectory
  # Hypothesis: There are different types of devices/device modes.
  # Some devices can only see a subset of the available bssid's
  bssid_id_map = {b: i for i, b in enumerate(list(bssid_grouped.keys()))}
  num_bssid = len(bssid_id_map)
  fn_grouped = dict(tuple(wifi.groupby("fn")))
  bssid_connected = np.zeros((num_bssid, num_bssid), dtype=np.bool)
  # Create a graph of what bssids co-occur with what other bssids within
  # the span of a trajectory
  fn_bssid_ids = {}
  for fn_id, fn in enumerate(list(fn_grouped.keys())):
    bssids = np.unique(fn_grouped[fn].bssid_wifi)
    bssid_ids = np.array([bssid_id_map[b] for b in bssids])
    fn_bssid_ids[fn] = bssid_ids
    bssid_connected[bssid_ids[:, None], bssid_ids] = True
  mean_bssid_connected = bssid_connected.mean(0)
    
  # g = nx.from_numpy_matrix(bssid_connected[:30, :30])
  # nx.draw(g)
  # plt.draw()
  # plt.savefig("graph.png", format="PNG")
  
  # # Compute the average network connection for each fn
  # fn_mean_connection = {
  #   fn: mean_bssid_connected[fn_bssid_ids[fn]].mean() for fn in list(
  #     fn_grouped.keys())}
  
  # For each waypoint: look at the closest observation from a different fn
  # to judge if a bssid is common / device specific
  waypoints_grouped = wifi.groupby([
    'fn', 't1_wifi', 'waypoint_interp_x', 'waypoint_interp_y']).size(
      ).reset_index()
  fns = waypoints_grouped.fn.values
  t1_wifis = waypoints_grouped.t1_wifi.values
  waypoint_positions = waypoints_grouped[[
    'waypoint_interp_x', 'waypoint_interp_y']].values
  rare_common_count = np.zeros((num_bssid, 2))
  for i in range(waypoints_grouped.shape[0]):
    num_wp_fn = (fns == fn).sum()
    fn = waypoints_grouped.fn[i]
    t1_wifi = waypoints_grouped.t1_wifi[i]
    distances = np.sqrt(
      ((waypoint_positions[i:i+1] - waypoint_positions)**2).sum(1)) + 9999*(
        fns == fn)
    nearest_id = np.argmin(distances)
    this_bssid = wifi.iloc[
      np.where((wifi.fn == fn) &  (wifi.t1_wifi == t1_wifi))[0]]
    nearest_bssid = wifi.iloc[
      np.where((wifi.fn == fns[nearest_id]) &  (wifi.t1_wifi == t1_wifis[
        nearest_id]))[0]]
    this_count = this_bssid.shape[0]
    nearest_count = nearest_bssid.shape[0]
    count_ratio_diff = max(this_count/nearest_count,nearest_count/this_count)-1
    if count_ratio_diff > 0.2:
      common_bssid = set(this_bssid.bssid_wifi).intersection(set(
        nearest_bssid.bssid_wifi))
      if (len(common_bssid)/min(this_count, nearest_count)) > 0.8:
        if this_count > nearest_count:
          rare_bssid = set(this_bssid.bssid_wifi).difference(set(
            common_bssid))
        else:
          rare_bssid = set(nearest_bssid.bssid_wifi).difference(set(
            common_bssid))
        rare_ids = np.array([bssid_id_map[b] for b in rare_bssid])
        common_ids = np.array([bssid_id_map[b] for b in common_bssid])
        rare_common_count[rare_ids, 0] += 1/num_wp_fn
        rare_common_count[common_ids, 1] += 1/num_wp_fn
        
  import pdb; pdb.set_trace()
  x=1

site_floors = df.iloc[df.test_site.values].groupby(
  ['site_id', 'text_level']).size().reset_index()
if debug_floor is not None:
  site_floors = site_floors.iloc[debug_floor:(debug_floor+1)]
sites = site_floors.site_id.values
floors = site_floors.text_level.values

if use_multiprocessing:
  floor_ids = np.arange(floors.size)
  with mp.Pool(processes=mp.cpu_count()-1) as pool:
    results = [pool.apply_async(
      store_floor_bssid_stats, args=(
        s, f, i, site_floors, data_folder, test_pred_fns, test_pred_times,
        test_pred_waypoints, chunk_ms, max_considered_near_distances,
        max_delay_ms, min_range_margin)) for (s, f, i) in zip(
          sites, floors, floor_ids)]
    all_results = [p.get() for p in results]
else:
  all_floor_waypoint_predictions = []
  for floor_id, (analysis_site, floor) in enumerate(zip(sites, floors)):
    store_floor_bssid_stats(
      analysis_site, floor, floor_id, site_floors, data_folder, test_pred_fns,
      test_pred_times, test_pred_waypoints, chunk_ms,
      max_considered_near_distances, max_delay_ms, min_range_margin)