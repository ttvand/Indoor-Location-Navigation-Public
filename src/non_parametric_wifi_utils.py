import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

import utils

def get_reference_scores(site_floors, reference_submission):
  reference_scores = np.zeros(site_floors.shape[0])
  for i in range(site_floors.shape[0]):
    site = site_floors.site_id.values[i]
    floor = site_floors.text_level.values[i]
    ref_score = reference_submission.error.values[
      (reference_submission.site == site) & (
        reference_submission.floor == floor)].mean()
    reference_scores[i] = ref_score
    
  return reference_scores

def multiple_floors_train_predict(
    config, df, debug_floor, reference_submission, use_multiprocessing,
    models_group_name, mode, holdout_df, test_floors, recompute_grouped_data,
    overwrite_models, test_type_mapping, only_public_test_preds,
    test_waypoint_times, store_all_wifi_predictions,
    store_full_wifi_predictions, debug_fn=None, verbose=True):
  data_folder = utils.get_data_folder()
  model_folder = data_folder.parent / 'Models' / models_group_name
  site_floors = df.iloc[df.test_site.values].groupby(
    ['site_id', 'text_level']).size().reset_index()
  if debug_floor is not None:
    site_floors = site_floors.iloc[debug_floor:(debug_floor+1)]
  sites = site_floors.site_id.values
  floors = site_floors.text_level.values
  ref_scores = get_reference_scores(site_floors, reference_submission) if (
    reference_submission) is not None else [None]*floors.size
  
  if use_multiprocessing:
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
      floor_ids = np.arange(floors.size)
      results = [pool.apply_async(
        floor_train_predict, args=(
          config, s, f, i, r, data_folder, model_folder, df, mode, holdout_df,
          test_floors, recompute_grouped_data, overwrite_models,
          test_type_mapping, only_public_test_preds, test_waypoint_times,
          store_all_wifi_predictions, store_full_wifi_predictions,
          debug_fn, verbose)) for (s, f, i, r) in zip(
            sites, floors, floor_ids, ref_scores)]
      all_outputs = [p.get() for p in results]
  else:
    all_outputs = []
    for floor_id, (analysis_site, floor, ref_score) in enumerate(zip(
        sites, floors, ref_scores)):
      all_outputs.append(
        floor_train_predict(
          config, analysis_site, floor, floor_id, ref_score, data_folder,
          model_folder, df, mode, holdout_df, test_floors,
          recompute_grouped_data, overwrite_models, test_type_mapping,
          only_public_test_preds, test_waypoint_times,
          store_all_wifi_predictions, store_full_wifi_predictions,
          debug_fn, verbose))
      
  return all_outputs

def floor_train_predict(
    config, analysis_site, floor, floor_id, ref_score, data_folder,
    model_folder, df, mode, holdout_df, test_floors, recompute_grouped_data,
    overwrite_models, test_type_mapping, only_public_test_preds,
    test_waypoint_times, store_all_wifi_predictions,
    store_full_wifi_predictions, debug_fn, verbose):
  test_preds = {}
  valid_preds = []
  all_wifi_predictions = []
  full_pos_preds = []
  floor_key = (analysis_site, floor)
  combined_full_pos_preds = {floor_key: None}
  
  site_model_folder = model_folder / analysis_site
  Path(site_model_folder).mkdir(parents=True, exist_ok=True)

  site_df = df[(df.site_id == analysis_site) & (df.num_wifi > 0)]
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
      return (test_preds, valid_preds, all_wifi_predictions,
              combined_full_pos_preds)
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
      return (test_preds, valid_preds, all_wifi_predictions,
              combined_full_pos_preds)
    valid = utils.load_site_floor(
      test_df_floor, recompute_grouped_data, test_floor=floor)
  else:
    train = utils.load_site_floor(
      floor_df[floor_df['mode'] == 'train'], recompute_grouped_data)
    valid = utils.load_site_floor(
      floor_df[floor_df['mode'] == 'valid'], recompute_grouped_data)

  # Train the wifi models
  utils.aggregate_wifi_near_time(train, config['time_range_max_strength'])
  utils.aggregate_wifi_near_time(valid, config['time_range_max_strength'])
  utils.interpolate_wifi_waypoints(
      train, recompute=True, batch_interpolated=True)
  bssid_grouped = utils.group_waypoints_bssid(train)
  model_type_prefix = 'test-' if mode == 'test' else ''
  model_path = site_model_folder / (model_type_prefix + floor + '.pickle')

  if model_path.exists() and not overwrite_models:
    with open(model_path, 'rb') as f:
      model = pickle.load(f)
  else:
    model = fit_model(train, bssid_grouped, config, data_folder)

    with open(model_path, 'wb') as handle:
      pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # Generate predictions with the wifi model
  make_predict_efforts = [v['file_meta']['mode'] != 'test' or (
    test_type_mapping[v['file_meta'].fn] == 'public') or (
      not only_public_test_preds) for v in valid]
  wifi_pos_preds = []
  for j, v in enumerate(valid):
    # Locate all unique wifi time observations
    if debug_fn is not None:
      if v['file_meta'].fn == debug_fn:
        import pdb; pdb.set_trace()
        x=1
      else:
        continue
    pos_pred, full_pos_pred = predict_trajectory(
      v, make_predict_efforts[j], model, store_full_wifi_predictions, config)
    wifi_pos_preds.append(pos_pred)
    if store_full_wifi_predictions and isinstance(full_pos_pred, pd.DataFrame):
      if len(full_pos_preds):
        full_pos_pred.drop(['x', 'y'], axis=1, inplace=True)
      full_pos_preds.append(full_pos_pred)
  if store_full_wifi_predictions and full_pos_preds:
    combined_full_pos_preds[floor_key] = pd.concat(full_pos_preds, axis=1)
      
  all_preds_floor = []
  all_actual_floor = []
  for i, v in enumerate(valid):
    # Interpolate the locations of the unique wifi time observations
    waypoint_times = test_waypoint_times[v['file_meta'].ext_path[5:-4]] if (
        mode == 'test') else v['waypoint'].time.values
    v_preds = utils.interpolate_predictions(
      wifi_pos_preds[i], waypoint_times)
    if mode == 'test':
      for waypoint_id in range(waypoint_times.shape[0]):
        test_preds[analysis_site, v['file_meta'].fn, waypoint_times[
          waypoint_id]] = (numeric_floor, v_preds[waypoint_id, 0],
                            v_preds[waypoint_id, 1])
    else:
      if store_all_wifi_predictions:
        pos_preds = wifi_pos_preds[i]
        for k in pos_preds:
          
          all_wifi_predictions.append({
            'site': analysis_site,
            'fn': v['file_meta'].fn,
            'time': k,
            'floor': floor,
            'numeric_floor': numeric_floor,
            'x_pred': pos_preds[k][0],
            'y_pred': pos_preds[k][1],
          })
      
      all_preds_floor.append(v_preds)
      actuals = v['waypoint'].iloc[:, 2:].values
      all_actual_floor.append(actuals)
      
      for waypoint_id in range(waypoint_times.shape[0]):
        squared_error = (
          (v_preds[waypoint_id] - actuals[waypoint_id])**2).sum()
        valid_preds.append({
          'site': analysis_site,
          'fn': v['file_meta'].fn,
          'waypoint_time': waypoint_times[waypoint_id],
          'floor': floor,
          'numeric_floor': numeric_floor,
          'x_pred': v_preds[waypoint_id, 0],
          'y_pred': v_preds[waypoint_id, 1],
          'x_actual': v['waypoint'].x_waypoint[waypoint_id],
          'y_actual': v['waypoint'].y_waypoint[waypoint_id],
          'squared_error': squared_error,
          'error': np.sqrt(squared_error),
          })
  
  if mode != 'test' and verbose and ref_score is not None:
    all_preds_floor = np.concatenate(all_preds_floor)
    all_actual_floor = np.concatenate(all_actual_floor)
    floor_loss = utils.get_loss(all_preds_floor, all_actual_floor)
    ref_score_change = floor_loss-ref_score
    print(f"{floor_id} {analysis_site} {floor} loss: {floor_loss:.2f}\
 ({ref_score_change:.2f}) - {all_preds_floor.size}")
  elif verbose:
    print(f"{floor_id} {analysis_site} {floor}")
    
  return test_preds, valid_preds, all_wifi_predictions, combined_full_pos_preds

def get_all_floor_preds(
    mode, config, site, site_id, data_folder, model_folder, df,
    analysis_floors, debug_fn, verbose=True):
  site_floors = np.sort(np.unique(df.text_level[
    (df.site_id == site) & (df['mode'] != 'test')].values))
  
  if debug_fn is not None and df.site_id.values[
      np.where(df.fn.values == debug_fn)[0][0]] != site:
    return None
  
  # Load the wifi models for all site floors
  models = {}
  for floor in site_floors:
    mode_prefix = 'test-' if mode == 'test' else ''
    model_path = model_folder / site / (mode_prefix + floor + '.pickle')
  
    with open(model_path, 'rb') as f:
      models[floor] = pickle.load(f)
  
  # Load all prediction inputs
  trajectories = []
  for floor_id, floor in enumerate(site_floors):
    if verbose:
      print(f"Load floor {floor_id+1} of {site_floors.size}")
    numeric_floor = utils.TEST_FLOOR_MAPPING[floor]
    site_df = df[(df.site_id == site) & (df.num_wifi > 0)]
    analysis_df = site_df[(site_df['mode'] == mode)]
    
    target_floors = np.array(
      [analysis_floors[fn] for fn in analysis_df['fn'].values])
    correct_floor = target_floors == numeric_floor
    analysis_df_floor = analysis_df[correct_floor]
    
    if analysis_df_floor.shape[0] > 0:
      test_floor = floor if mode =='test' else None
      trajectories.extend(utils.load_site_floor(
        analysis_df_floor, recompute_grouped_data=False,
        test_floor=test_floor))
  
  # Generate predictions for all trajectories for all floors
  make_predict_efforts = [True for t in trajectories]
  analysis_preds = []
  for j, t in enumerate(trajectories):
    if verbose:
      print(f"Trajectory {j+1} of {len(trajectories)}")
    fn = t['file_meta'].fn
    
    if (debug_fn is not None) and fn != debug_fn:
      continue
    
    debug_floor_distances = {}
    for floor in site_floors:
      # Locate all unique wifi time observations
      _, full_pos_pred = predict_trajectory(
        t, make_predict_efforts[j], models[floor], True, config)
      
      distances = full_pos_pred.values[:, 2:]
      min_distances = distances.min(0)
      
      if (debug_fn is not None) and fn == debug_fn:
        print(debug_fn, floor)
        debug_floor_distances[floor] = min_distances
      
      analysis_preds.append({
        'site': site,
        'floor': floor,
        'numeric_floor': utils.TEST_FLOOR_MAPPING[floor],
        'reference_floor_label': analysis_floors[fn],
        'fn': fn,
        'min_min_distance': min_distances.min(),
        'mean_min_distance': min_distances.mean(),
        'max_min_distance': min_distances.max(),
        'min_distance_q0.1': np.quantile(min_distances, 0.1),
        'min_distance_q0.2': np.quantile(min_distances, 0.2),
        'min_distance_q0.3': np.quantile(min_distances, 0.3),
        'min_distance_q0.4': np.quantile(min_distances, 0.4),
        'min_distance_q0.5': np.quantile(min_distances, 0.5),
        'min_distance_q0.6': np.quantile(min_distances, 0.6),
        'min_distance_q0.7': np.quantile(min_distances, 0.7),
        'min_distance_q0.8': np.quantile(min_distances, 0.8),
        'min_distance_q0.9': np.quantile(min_distances, 0.9),
        })
      
    if (debug_fn is not None) and fn == debug_fn:
      combined_debug = pd.DataFrame(debug_floor_distances)
      debug_save_path = utils.get_data_folder() / (
        "all_floor_wifi_distances " + debug_fn + '.csv')
      combined_debug.to_csv(debug_save_path, index=True)
  
  print(f"Done with site {site_id+1} of 24: {site}")
    
  return pd.DataFrame(analysis_preds)

def fit_model(trajectories, bssid_grouped, config, data_folder):
  min_train_points = config['min_train_points']
  min_train_fns = config['min_train_fns']
  unique_model_frequencies = config['unique_model_frequencies']
  limit_train_near_waypoints = config['limit_train_near_waypoints']
  
  if limit_train_near_waypoints:
    all_waypoints = np.concatenate([
        t['wifi_waypoints'].loc[t['wifi_waypoints'].near_waypoint_t1, [
          'waypoint_interp_x', 'waypoint_interp_y']].values
        for t in trajectories
    ])
  else:
    all_waypoints = np.concatenate([
        t['wifi_waypoints'][['waypoint_interp_x', 'waypoint_interp_y']].values
        for t in trajectories
    ])
  unique_waypoints = np.unique(all_waypoints, axis=0)
  unique_bssid = np.sort(
      np.unique(
          np.concatenate([
              t['wifi_waypoints']['bssid_wifi'].values for t in trajectories
          ])))
  num_waypoints = unique_waypoints.shape[0]
  num_bssid = unique_bssid.size
  unique_waypoints_map = {
      k: (unique_waypoints[k, 0], unique_waypoints[k, 1])
      for k in range(num_waypoints)
  }
  unique_waypoints_inv_map = {v: k for k, v in unique_waypoints_map.items()}
  unique_bssid_inv_map = {unique_bssid[k]: k for k in range(num_bssid)}

  # Map interpolated waypoints to wifi t1 times and source fns
  t1_waypoints = {}
  waypoint_fns = {}
  for t in trajectories:
    time_pos = t['wifi_waypoints'].groupby('t1_wifi').first().reset_index()
    for i in range(time_pos.shape[0]):
      k = (time_pos.waypoint_interp_x[i], time_pos.waypoint_interp_y[i])
      vt = time_pos.t1_wifi[i]
      vfn = t['file_meta'].fn
      if k in t1_waypoints:
        t1_waypoints[k].append(vt)
        waypoint_fns[k].append(vfn)
      else:
        t1_waypoints[k] = [vt]
        waypoint_fns[k] = [vfn]

  strengths = np.ones((num_waypoints, num_bssid), dtype=np.int16)
  delays = -np.ones((num_waypoints, num_bssid), dtype=np.float32)
  num_fns = np.full(num_bssid, np.nan)

  for b_id, b in enumerate(bssid_grouped):
    bssid_id = unique_bssid_inv_map[b]
    wifi_location = bssid_grouped[b]
    if limit_train_near_waypoints:
      wifi_location = wifi_location.iloc[wifi_location.near_waypoint_t1.values]
    # assert np.all((np.diff(wifi_location.t2_wifi) > 0)
    #               | np.diff(wifi_location.file_id) != 0)

    # Duplicate signals with a difference of 1 ms are asserted to have the same
    # rssid
    rss = wifi_location.rssid_wifi.values
    # dup_ids = np.where((np.diff(wifi_location.t2_wifi) == 1)
    #                    & (rss[:-1] == rss[1:]))[0]
    frequencies = np.unique(wifi_location.freq_wifi.values)

    if unique_model_frequencies and frequencies.size > 1:
      print("Different frequencies, ignoring model")
      continue
    keep_rows = np.ones_like(rss, dtype=bool)
    
    # Commented - Also keep duplicate measurements!
    # keep_rows[dup_ids + 1] = False 
    if keep_rows.sum() < max(1, min_train_points-max(
        0, 10-0.5*len(trajectories))):
      continue
    
    num_fn = np.unique(wifi_location.file_id.values).size
    if num_fn < min(0.2*len(trajectories), min_train_fns):
      continue
      
    keep_wifi_location = wifi_location.iloc[keep_rows]
    keep_rss = keep_wifi_location.rssid_wifi.values
    keep_x = keep_wifi_location.waypoint_interp_x.values
    keep_y = keep_wifi_location.waypoint_interp_y.values
    keep_delays = keep_wifi_location.most_recent_t2_wifi.values - (
        keep_wifi_location.t2_wifi.values)
    rows = np.array(
        [unique_waypoints_inv_map[(x, y)] for (x, y) in zip(keep_x, keep_y)])

    assert np.all(strengths[rows, bssid_id] > 0)
    assert np.all(delays[rows, bssid_id] < 0)

    strengths[rows, bssid_id] = keep_rss
    delays[rows, bssid_id] = keep_delays
    num_fns[bssid_id] = num_fn

  # Drop bssids that were filtered out
  keep_bssid_cols = ~np.isnan(num_fns)
  strengths = strengths[:, keep_bssid_cols]
  delays = delays[:, keep_bssid_cols]
  unique_bssid = unique_bssid[keep_bssid_cols]
  unique_bssid_inv_map = {unique_bssid[k]: k for k in range(unique_bssid.size)}
  num_fns = num_fns[keep_bssid_cols].astype(np.int64)

  # missing_distances = np.zeros_like(delays)
  # missing_distances[strengths < 0] = ((strengths[strengths < 0] + 100)**2)/(
  #   100)*(0.75**((np.maximum(1000, delays[strengths < 0])-1000)/1000))

  # Extract the number of non delayed observations
  # floor_stats_wifi_path = data_folder / 'train' / t['file_meta'].site_id / (
  #   t['file_meta'].text_level) / 'bssid_stats.csv'
  # floor_bssid_stats = pd.read_csv(floor_stats_wifi_path)
  # non_delayed_map = {k: v for k, v in zip(
  #   floor_bssid_stats.bssid, floor_bssid_stats.num_non_delayed)}
  # non_delayed_counts = np.array([non_delayed_map[b] for b in unique_bssid])
  non_delayed_counts = np.ones_like(unique_bssid)

  if len(unique_bssid_inv_map) == 0:
    import pdb; pdb.set_trace()
    x=1

  model = {
      'waypoints': unique_waypoints_map,
      'bssid': unique_bssid_inv_map,
      't1_waypoints': t1_waypoints,
      'waypoint_fns': waypoint_fns,
      'strengths': strengths,
      'num_bssid_per_waypoint': (strengths < 0).sum(1),
      'delays': delays,
      # 'missing_distances': missing_distances,
      'non_delayed_counts': non_delayed_counts,
      'num_fns': num_fns,
  }
  
  return model

def distances_v1(model, config, rssid, time_offsets, cols):
  strengths = model['strengths']
  delays = model['delays']
  missing_distances = model['missing_distances']
  
  num_bssid = len(model['bssid'])
  missing_cols = np.setdiff1d(np.arange(num_bssid), cols)
  
  # Compute the distance for the columns where this bssid occurs and where
  # the reference bssid occurs
  bools = np.zeros_like(strengths, dtype=np.bool)
  bools[:, cols] = True
  bools &= (strengths <= 0)
  all_distances = ((strengths[:, cols] - np.expand_dims(rssid, 0))**2)/(
    100)*(
      0.75**((np.maximum(1000, np.expand_dims(time_offsets, 0))-1000)/1000))*(
        0.75**((np.maximum(1000, delays[:, cols])-1000)/1000))
  shared_distances = (all_distances*bools[:, cols]).sum(1)
  
  
  # Compute the distance for the columns where this bssid does not occur and
  # where the reference bssid occurs
  my_missing_distances = missing_distances[:, missing_cols].sum(1)
  
  # Compute the distance for the columns where this bssid occurs, but the 
  # reference does not
  bools = np.zeros_like(strengths, dtype=np.bool)
  bools[:, cols] = True
  bools &= (strengths > 0)
  my_bssid_missing_distances = ((rssid + 100)**2)/(
    100)*(0.75**((np.maximum(1000, time_offsets)-1000)/1000))
  
  ref_missing_distances = (np.expand_dims(my_bssid_missing_distances, 0) * (
    bools[:, cols])).sum(1)
  
  distances = shared_distances + my_missing_distances + ref_missing_distances
  
  return distances

def distances_v2(model, config, rssid, time_offsets, cols, wifi_waypoints):
  strengths = model['strengths']
  delays = model['delays']
  delay_decay_exp_base = config['delay_decay_exp_base']
  
  num_bssid_per_waypoint = model['num_bssid_per_waypoint']
  num_cols = cols.size
  max_shared = np.maximum(20, np.minimum(num_cols, num_bssid_per_waypoint))
  
  # Compute the distance for the columns where this bssid occurs and where
  # the reference bssid occurs
  bools = np.zeros_like(strengths, dtype=np.bool)
  bools[:, cols] = True
  bools &= (strengths <= 0)
  shared_obs_count = bools.sum(1)
  shared_fractions = shared_obs_count/(max_shared)
  
  delay_factors = (delay_decay_exp_base**(
    (np.maximum(1000, np.expand_dims(time_offsets, 0))-1000)/1000))*(
        delay_decay_exp_base**((np.maximum(1000, delays[:, cols])-1000)/1000))
  delay_weights = (delay_factors*bools[:, cols]).sum(1)
  shared_distances = np.minimum(
    30, np.abs(strengths[:, cols] - np.expand_dims(rssid, 0)))
  shared_distances_weighted = (delay_factors*shared_distances*bools[
    :, cols]).sum(1)/(delay_weights+1e-9)
  shared_fraction_distances = (0.8-np.minimum(0.8, shared_fractions))*40
  
  distances = shared_fraction_distances + shared_distances_weighted
  
  # if wifi_waypoints is not None:
  #   greedy_pred = np.array(model['waypoints'][np.argmin(distances)])
  #   wifi_waypoints.index = np.arange(wifi_waypoints.shape[0])
  #   first_actual = wifi_waypoints.loc[0, [
  #     'waypoint_interp_x', 'waypoint_interp_y']].values
  #   error = np.sqrt(((greedy_pred-first_actual)**2).sum())
  #   if error > 30:
  #     import pdb; pdb.set_trace()
  #     x=1
  
  return distances

def get_shared_fractions(
    config, strengths, delays, rssid, time_offsets, num_fns, cols, bools):
  delay_decay_penalty_exp_base = config['delay_decay_penalty_exp_base']
  inv_fn_count_penalty_exp = config['inv_fn_count_penalty_exp']
  
  inv_fn_count_all_weights = (1/num_fns)**inv_fn_count_penalty_exp
  inv_fn_count_weight = inv_fn_count_all_weights[cols]
  
  # Shared weights
  shared_bssid_weights_ref = (bools[:, cols]*(strengths[:, cols]+100)*(
    delay_decay_penalty_exp_base**((
      np.maximum(1000, delays[:, cols])-1000)/1000)) * np.expand_dims(
        inv_fn_count_weight, 0)).sum(1)
  shared_bssid_weights_inf = ((rssid+100)*(
    delay_decay_penalty_exp_base**((
      np.maximum(1000, time_offsets)-1000)/1000))*(inv_fn_count_weight)).sum()
  
  # Inference not reference weights
  bools = np.zeros_like(strengths, dtype=np.bool)
  bools[:, cols] = True
  bools &= (strengths > 0)
  inf_not_ref_weights = (bools[:, cols]*(np.expand_dims(rssid, 0)+100)*(
    delay_decay_penalty_exp_base**((
      np.maximum(1000, np.expand_dims(time_offsets, 0))-1000)/1000))*(
        np.expand_dims(inv_fn_count_weight, 0))).sum(1)
  
  # Reference not inference weights
  bools = strengths < 0
  bools[:, cols] = False
  ref_not_inf_weights = (bools*(strengths+100)*(
    delay_decay_penalty_exp_base**((
      np.maximum(1000, delays)-1000)/1000))*(
        np.expand_dims(inv_fn_count_all_weights, 0))).sum(1)
  
  # Combine the weights
  numerator = shared_bssid_weights_ref + shared_bssid_weights_inf
  denominator = numerator + inf_not_ref_weights + ref_not_inf_weights
  
  return numerator/denominator

def distances_v3(model, config, rssid, time_offsets, cols, wifi_waypoints):
  strengths = model['strengths']
  delays = model['delays']
  non_delayed_counts = model['non_delayed_counts']
  num_fns = model['num_fns']
  
  delay_decay_exp_base = config['delay_decay_exp_base']
  non_shared_penalty_start = config['non_shared_penalty_start']
  non_shared_penalty_exponent = config['non_shared_penalty_exponent']
  non_shared_penalty_constant = config['non_shared_penalty_constant']
  inv_fn_count_distance_exp = config['inv_fn_count_distance_exp']
  
  # Compute the shared weighted distance
  bools = np.zeros_like(strengths, dtype=np.bool)
  bools[:, cols] = True
  bools &= (strengths <= 0)
  
  delay_factor = (delay_decay_exp_base**(
    (np.maximum(1000, np.expand_dims(time_offsets, 0))-1000)/1000))*(
        delay_decay_exp_base**((np.maximum(1000, delays[:, cols])-1000)/1000))
  # wifi_count_factor = np.expand_dims(1/((non_delayed_counts[cols] + 3)**0.5), 0)
  wifi_count_factor = np.expand_dims(np.ones_like(non_delayed_counts[cols]), 0)
  rare_fn_factor = np.expand_dims(
    (1/num_fns[cols])**inv_fn_count_distance_exp, 0)
  shared_weights = rare_fn_factor*wifi_count_factor*(
    delay_factor)*bools[:, cols]
  shared_weights_normalized = shared_weights.sum(1)
  shared_distances = np.abs(strengths[:, cols] - np.expand_dims(rssid, 0))
  shared_distances_weighted = ((shared_weights*shared_distances*bools[
    :, cols]).sum(1))/(shared_weights_normalized+1e-9)
  
  shared_fractions = get_shared_fractions(
    config, strengths, delays, rssid, time_offsets, num_fns, cols, bools)
  
  # import pdb; pdb.set_trace()
  # 'non_shared_penalty_start': 0.9,
  # 'non_shared_penalty_exponent': 0.5,
  # 'non_shared_penalty_constant': 30,
  # shared_fraction_penalties = (0.8-np.minimum(0.8, shared_fractions))*40
  shared_fraction_penalties = non_shared_penalty_constant*(np.maximum(
    0, non_shared_penalty_start-shared_fractions)**(
      non_shared_penalty_exponent))
  
  distances = shared_distances_weighted + shared_fraction_penalties
  # distances = alpha*shared_distances_weighted + (
  #   1-alpha)*non_shared_distances_weighted
  
  return distances

def triangulate(wifi_location, model, store_full_wifi_predictions, config,
                wifi_waypoints, return_distances=False):
  bssid_col_mapping = model['bssid']
  # Ignore bssid's not seen during training
  training_bssid = np.array([b in bssid_col_mapping for b in (
    wifi_location.bssid_wifi)])
  wifi_location = wifi_location.iloc[training_bssid]
  t2 = wifi_location.t2_wifi.values
  if t2.size == 0:
    # No bssids are seen in training - make a default high distance prediction
    positions = np.array(list(model['waypoints'].values()))
    x = positions[:, 0]
    y = positions[:, 1]
    distances = 999*np.ones_like(x)
    full_pos_pred = pd.DataFrame(zip(x, y, distances))
    full_pos_pred.columns = ['x', 'y', 'distance']
    triangulate_result = np.array(model['waypoints'][np.argmin(distances)])
  else:
    most_recent_t2 = t2.max()
    time_offsets = most_recent_t2 - t2
    rssid = wifi_location.rssid_wifi.values
    cols = np.array([bssid_col_mapping[b] for b in wifi_location.bssid_wifi])
  
    distances = distances_v3(
      model, config, rssid, time_offsets, cols, wifi_waypoints)
    triangulate_result = np.array(model['waypoints'][np.argmin(distances)])
  
    if store_full_wifi_predictions:
      positions = np.array(list(model['waypoints'].values()))
      x = positions[:, 0]
      y = positions[:, 1]
      full_pos_pred = pd.DataFrame(zip(x, y, distances))
      full_pos_pred.columns = ['x', 'y', 'distance']
    else:
      full_pos_pred = None

  if return_distances:
    return distances
  else:
    return triangulate_result, full_pos_pred

def predict_trajectory(
    t, make_effort, model, store_full_wifi_predictions, config):
  # Locate all unique wifi time observations
  wifi_groups = dict(tuple(t['wifi'].groupby('t1_wifi')))
  if 'wifi_waypoints' in t:
    wifi_waypoint_groups = dict(tuple(t['wifi_waypoints'].groupby('t1_wifi')))
  else:
    wifi_waypoint_groups = {}
  if not make_effort:
    wifi_pos_preds = {k: np.zeros(2) for k in wifi_groups}
    full_pos_preds = []
  else:
    wifi_pos_preds = {}
    full_pos_preds = []
    for i, k in enumerate(wifi_groups):
      wifi_pos_preds[k], full_pos_pred = triangulate(
        wifi_groups[k], model, store_full_wifi_predictions, config,
        wifi_waypoint_groups.get(k, None))
      if store_full_wifi_predictions:
        full_pos_pred.rename({'distance': t['file_meta'].fn + '_' + str(k)},
                             axis=1, inplace=True)
        if i > 0:
          full_pos_pred.drop(['x', 'y'], axis=1, inplace=True)
        full_pos_preds.append(full_pos_pred)
      
    if store_full_wifi_predictions:
      full_pos_preds = pd.concat(full_pos_preds, axis=1)
      
  return wifi_pos_preds, full_pos_preds
