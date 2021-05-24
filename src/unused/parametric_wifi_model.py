import datetime
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import utils
from scipy.optimize import minimize

# Model idea: Maximize x, y, signal strength and central stength for all
# physical access points (bssid) - This allows for a very simple loss function
# that can be optimized for triangulation on the holdout data
models_group_name = 'parametric_wifi'
overwrite_models = not True
recompute_grouped_data = not True
min_train_points = 10
max_error_clip = 50
unique_model_frequencies = False
mode = ['valid', 'test'][0]


def predict(x, fit_x, fit_y):
  squared_distances = (x[0] - fit_x)**2 + (x[1] - fit_y)**2
  expected_rss = x[2] - squared_distances / 20

  return expected_rss


def train_loss(x, fit_rss, fit_x, fit_y):
  expected_rss = predict(x, fit_x, fit_y)
  return np.sum(np.abs(expected_rss - fit_rss))


def fit_model(wifi_location, min_train_points, unique_model_frequencies,
              fit_id):
  assert np.all((np.diff(wifi_location.t2_wifi) > 0)
                | np.diff(wifi_location.file_id) != 0)
  # Duplicate signals with a difference of 1 ms are asserted to have the same
  # rssid
  rss = wifi_location.rssid_wifi.values
  dup_ids = np.where((np.diff(wifi_location.t2_wifi) == 1)
                     & (rss[:-1] == rss[1:]))[0]
  frequencies = np.unique(wifi_location.freq_wifi.values)

  if unique_model_frequencies and frequencies.size > 1:
    print("Different frequencies, ignoring model")
    return None
  keep_rows = np.ones_like(rss, dtype=bool)
  keep_rows[dup_ids + 1] = False

  if keep_rows.sum() < min_train_points:
    return None

  fit_rss = rss[keep_rows]
  fit_x = wifi_location.waypoint_interp_x.values[keep_rows]
  fit_y = wifi_location.waypoint_interp_y.values[keep_rows]

  max_id = np.argmax(fit_rss)
  x0 = np.array([fit_x[max_id], fit_y[max_id], fit_rss[max_id]])

  data_loss = partial(train_loss, fit_rss=fit_rss, fit_x=fit_x, fit_y=fit_y)
  res = minimize(
      data_loss,
      x0,
      method='nelder-mead',
      options={
          'xatol': 1e-8,
          'disp': False
      })

  model = {
      'frequencies': frequencies,
      'num_points': fit_rss.size,
      'model_params': res.x,
      'model_success': res.success,
  }

  return model


def triangulate(wifi_location,
                models,
                max_error_clip,
                unique_model_frequencies,
                ignore_offset_s=2):
  t2 = wifi_location.t2_wifi.values
  most_recent_t2 = t2.max()
  time_offsets = (most_recent_t2 - t2) / 1000
  recent_ids = time_offsets <= ignore_offset_s
  recent_location = wifi_location[recent_ids]
  valid_models = [
      b in models and (not unique_model_frequencies or
                       (models[b]['frequencies'][0] == f))
      for (b, f) in zip(recent_location.bssid_wifi, recent_location.freq_wifi)
  ]
  triangulate_location = recent_location[valid_models]

  # print(wifi_location.shape[0], recent_location.shape[0],
  #       triangulate_location.shape[0])
  model_keys = triangulate_location.bssid_wifi.values
  if not model_keys.size:
    return np.stack([models[k]['model_params'] for k in models]).mean(0)[:2]
  model_sizes = np.array([models[k]['num_points'] for k in model_keys])
  model_params = np.stack([models[k]['model_params'] for k in model_keys])

  model_recency_weights = 0.8**time_offsets[recent_ids][valid_models]
  model_size_weights = np.minimum(100, model_sizes)
  actual_rss = triangulate_location.rssid_wifi.values
  signal_strength_weights = np.minimum(1, 2**((actual_rss + 50) / 10))
  combined_weights = model_recency_weights * model_size_weights * (
      signal_strength_weights)

  def fit_loss(x,
               model_params,
               combined_weights,
               actual_rss,
               max_error_clip=max_error_clip):
    expected_rss = predict(model_params, x[0], x[1])
    return (np.abs(expected_rss - actual_rss) * combined_weights).sum()

  x0 = np.array([model_params[:, 0].mean(), model_params[:, 1].mean()])

  data_loss = partial(
      fit_loss,
      model_params=np.transpose(model_params),
      combined_weights=combined_weights,
      actual_rss=actual_rss)
  res = minimize(
      data_loss,
      x0,
      method='nelder-mead',
      options={
          'xatol': 1e-8,
          'disp': False
      })

  # if not res.success:
  #   import pdb; pdb.set_trace()
  #   x=1

  return res.x


data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
stratified_holdout_path = data_folder / 'holdout_ids.csv'
model_folder = data_folder.parent / 'Models' / models_group_name
if not 'df' in locals() or not 'holdout_df' in locals() or (
    not 'test_waypoint_times' in locals()) or not 'test_floors' in locals():
  df = pd.read_csv(summary_path)
  holdout_df = pd.read_csv(stratified_holdout_path)
  test_waypoint_times = utils.get_test_waypoint_times(data_folder)
  test_floors = utils.get_test_floors(data_folder)

aggregate_scores = np.zeros((len(utils.TEST_SITES), 2))
test_preds = {}
for analysis_site_id, analysis_site in enumerate(utils.TEST_SITES):
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
  floors = sorted(set([l for l in site_df.text_level if not pd.isnull(l)]))
  all_losses = np.zeros((len(floors), 2))
  for floor_id, floor in enumerate(floors):
    floor_df = site_df[site_df.text_level == floor]
    if mode == 'test':
      target_floors = np.array([test_floors[fn] for fn in test_df['fn'].values])
      numeric_floor = utils.TEST_FLOOR_MAPPING[floor]
      correct_test_floor = target_floors == numeric_floor
      if not np.any(correct_test_floor):
        continue
      test_df_floor = test_df[correct_test_floor]

    # import pdb; pdb.set_trace()
    num_train_waypoints = floor_df.num_train_waypoints[floor_df['mode'] ==
                                                       'train'].sum()
    num_valid_waypoints = floor_df.num_train_waypoints[floor_df['mode'] ==
                                                       'valid'].sum()
    print(floor, num_train_waypoints, num_valid_waypoints)

    # if (analysis_site, floor) == ('5d2709a003f801723c3251bf', '3F'):
    #   import pdb; pdb.set_trace()

    # Load the combined floor train data
    if mode == 'test':
      with pd.option_context("mode.chained_assignment", None):
        floor_df.loc[:, 'mode'] = 'all_train'
      train = utils.load_site_floor(floor_df, recompute_grouped_data)
      valid = utils.load_site_floor(test_df_floor, recompute_grouped_data)
    else:
      train = utils.load_site_floor(floor_df[floor_df['mode'] == 'train'],
                                    recompute_grouped_data)
      valid = utils.load_site_floor(floor_df[floor_df['mode'] == 'valid'],
                                    recompute_grouped_data)

    unique_bssid_train = np.sort(
        np.unique(np.concatenate([t['wifi'].bssid_wifi for t in train])))
    unique_bssid_valid = np.sort(
        np.unique(np.concatenate([v['wifi'].bssid_wifi for v in valid])))

    # Train the wifi models
    utils.interpolate_wifi_waypoints(train)
    bssid_grouped = utils.group_waypoints_bssid(train)
    # group_sizes = np.array([bssid_grouped[b].shape[0] for b in bssid_grouped])
    model_type_prefix = 'test-' if mode == 'test' else ''
    models_path = site_model_folder / (model_type_prefix + floor + '.pickle')
    models = {}
    if models_path.exists() and not overwrite_models:
      with open(models_path, 'rb') as f:
        models = pickle.load(f)

    for i, k in enumerate(list(bssid_grouped.keys())):
      if not k in models or overwrite_models:
        # print(i)
        models[k] = fit_model(bssid_grouped[k], min_train_points,
                              unique_model_frequencies, i)

    with open(models_path, 'wb') as handle:
      pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    valid_models = {
        k: v
        for k, v in models.items()
        if v is not None and (v['model_success'])
    }

    # Generate predictions with the wifi models
    all_preds_floor = []
    all_actual_floor = []
    for i, v in enumerate(valid):
      # print(i)
      # Locate all unique wifi time observations
      wifi_groups = dict(tuple(v['wifi'].groupby('t1_wifi')))
      wifi_pos_preds = {
          k: triangulate(wifi_groups[k], valid_models, max_error_clip,
                         unique_model_frequencies) for k in wifi_groups
      }

      # Interpolate the locations of the unique wifi time observations
      waypoint_times = test_waypoint_times[v['file_meta'].ext_path[5:-4]] if (
          mode == 'test') else v['waypoint'].time.values
      v_preds = utils.interpolate_predictions(wifi_pos_preds, waypoint_times)
      if mode == 'test':
        for waypoint_id in range(waypoint_times.shape[0]):
          test_preds[analysis_site, v['file_meta'].fn,
                     waypoint_times[waypoint_id]] = (numeric_floor,
                                                     v_preds[waypoint_id, 0],
                                                     v_preds[waypoint_id, 1])
      else:
        all_preds_floor.append(v_preds)
        all_actual_floor.append(v['waypoint'].iloc[:, 2:].values)

    if mode != 'test':
      all_preds_floor = np.concatenate(all_preds_floor)
      all_actual_floor = np.concatenate(all_actual_floor)
      floor_loss = utils.get_loss(all_preds_floor, all_actual_floor)
      all_losses[floor_id] = (floor_loss, all_actual_floor.shape[0])

      print(f"{floor} loss: {floor_loss:.2f}")

  if mode != 'test':
    site_num_obs = all_losses[:, 1].sum()
    weighted_loss = (all_losses[:, 0] * all_losses[:, 1]).sum()
    site_loss = weighted_loss / site_num_obs
    print(f"Site {analysis_site} ({analysis_site_id+1}) loss: {site_loss:.2f}")
    aggregate_scores[analysis_site_id] = (site_loss, site_num_obs)

if mode == 'test':
  submission = utils.convert_to_submission(data_folder, test_preds)
  submission_folder = data_folder / "submissions"
  record_time = str(datetime.datetime.now())[:19]
  submission_ext = models_group_name + ' ' + record_time + '.csv'
  submission.to_csv(submission_folder / submission_ext, index=False)
else:
  relative_weights = np.sqrt(aggregate_scores[:, 1])
  relative_weights /= np.sum(relative_weights)
  holdout_loss = (relative_weights * aggregate_scores[:, 0]).sum()
  print(f"Holdout aggregate loss: {holdout_loss:.2f}")
