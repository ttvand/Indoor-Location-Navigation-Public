import numpy as np
import pandas as pd
import pickle

import utils

def run():
  print("Combining sensor models with device ids")
  data_folder = utils.get_data_folder()
  summary_path = data_folder / 'file_summary.csv'
  df = pd.read_csv(summary_path)
  model_folder = data_folder.parent / 'Models' / 'sensor_absolute_movement'
  absolute_fold_folder = model_folder / 'errors cv'
  distance_folder = model_folder.parent / 'sensor_distance'
  valid_path = model_folder / 'predictions' / 'relative_movement_v3_valid.csv'
  device_id_path = data_folder / 'device_ids.pickle'
  meta_sensor_path = data_folder / 'sensor_data' / 'meta.csv'
  
  with open(device_id_path, 'rb') as f:
    device_ids = pickle.load(f)
  meta_data = pd.read_csv(meta_sensor_path, dtype={'test_type': object})
  
  
  #################################################
  # A: Combine statistics at the trajectory level #
  #################################################
  device_ids_path = data_folder / 'inferred_device_ids.csv'
  if not device_ids_path.is_file():
    device_id_vals, device_drifts, device_id_merged_vals = zip(
      *list(device_ids.values()))
    device_ids_df = pd.DataFrame({
      'mode': [meta_data['mode'].values[np.where(meta_data.fn == fn)[0][
        0]] for fn in list(device_ids.keys())],
      'test_type': [meta_data['test_type'].values[np.where(meta_data.fn == fn)[
        0][0]] for fn in list(device_ids.keys())],
      'fn': list(device_ids.keys()),
      'device_id': list(device_id_vals),
      'device_id_drift': list(device_drifts),
      'device_id_merged': list(device_id_merged_vals),
      'site': [df.site_id.values[np.where(df.fn == fn)[0][0]] for fn in list(
        device_ids.keys())],
      'floor': [df.level.values[np.where(df.fn == fn)[0][0]] for fn in list(
        device_ids.keys())],
      'start_time': [meta_data.start_time.values[np.where(meta_data.fn == fn)[
        0][ 0]] for fn in list(device_ids.keys())],
      'end_time': [meta_data.end_time.values[np.where(meta_data.fn == fn)[0][
        -1]] for fn in list(device_ids.keys())],
      'first_last_wifi_time': [df.first_last_wifi_time.values[
        np.where(df.fn == fn)[0][0]] for fn in list(device_ids.keys())],
      })
    device_ids_df.sort_values(['first_last_wifi_time', 'start_time'],
                              inplace=True)
    device_ids_df.to_csv(device_ids_path, index=False)
  
  
  ##############################################
  # B: Combine statistics at the segment level #
  ##############################################
  save_path_device_errors = data_folder / 'sensor_model_device_errors.csv'
  if not save_path_device_errors.is_file():
    train_preds_list = []
    for i in range(5):
      preds = pd.read_csv(absolute_fold_folder / f'preds_bag_fold_{i}.csv')
      preds['train_fold'] = i
      train_preds_list.append(preds)
      
    train_preds = pd.concat(train_preds_list)
      
    train_preds['mode'] = 'train'
    valid_preds = pd.read_csv(valid_path)
    valid_preds['train_fold'] = np.nan
    valid_preds['mode'] = 'valid'
    test_preds = meta_data[meta_data['mode'] == 'test']
    with pd.option_context('mode.chained_assignment', None):
      test_preds['x'] = np.nan
      test_preds['y'] = np.nan
      test_preds['x_pred'] = np.nan
      test_preds['y_pred'] = np.nan
      test_preds['train_fold'] = np.nan
      test_preds.rename(columns={"level": "floor"}, inplace=True)
      test_preds = test_preds.loc[:, valid_preds.columns]
    
    all_preds = pd.concat([train_preds, valid_preds, test_preds])
    
    all_preds['device_id'] = -3
    all_preds['device_id_drift'] = False
    all_preds['device_id_merged'] = -3
    all_preds['error'] = -2
    all_preds['start_time'] = -1
    all_preds['end_time'] = -1
    all_preds.index = np.arange(all_preds.shape[0])
    for i in range(all_preds.shape[0]):
      print(i)
      fn = all_preds.fn.values[i]
      mode = all_preds['mode'].values[i]
      if mode == 'test':
        error = np.nan
      else:
        error = np.sqrt((all_preds.x.values[i]-all_preds.x_pred.values[i])**2 + (
          all_preds.y.values[i]-all_preds.y_pred.values[i])**2)
      sub_trajectory_id = all_preds.sub_trajectory_id.values[i]
      meta_row = np.where((meta_data.fn.values == fn) & (
        meta_data.sub_trajectory_id.values == sub_trajectory_id))[0][0]
      start_time = meta_data.start_time.values[meta_row]
      end_time = meta_data.end_time.values[meta_row]
      df_row = np.where(df.fn == fn)[0][0]
      first_last_wifi_time = df.first_last_wifi_time.values[df_row]
      if np.isnan(first_last_wifi_time):
        assert mode == 'train'
        first_last_wifi_time = df.start_time.values[df_row]
      
      all_preds.loc[i, 'device_id'] = device_ids[fn][0]
      all_preds.loc[i, 'device_id_drift'] = device_ids[fn][1]
      all_preds.loc[i, 'device_id_merged'] = device_ids[fn][2]
      all_preds.loc[i, 'error'] = error
      all_preds.loc[i, 'start_time'] = start_time
      all_preds.loc[i, 'end_time'] = end_time
      all_preds.loc[i, 'first_last_wifi_time'] = first_last_wifi_time
      
    all_preds.sort_values([
      'device_id', 'first_last_wifi_time', 'sub_trajectory_id'], inplace=True)
    all_preds.to_csv(save_path_device_errors, index=False)
  
  save_path_fn_errors = data_folder / "fn_device_errors.csv"
  if not save_path_fn_errors.is_file():
    device_errors = pd.read_csv(save_path_device_errors)
    device_errors.sort_values(
      ['site', 'fn', 'sub_trajectory_id'], inplace=True)
    device_errors.index = np.arange(device_errors.shape[0])
    device_errors['new_device_id'] = [
      True] + (device_errors.device_id.values[:-1] != (
        device_errors.device_id.values[1:])).tolist()
    device_errors['dist'] = np.sqrt(
      device_errors.x.values**2 + device_errors.y.values**2)
    distance_cv_folder = distance_folder / 'distance_cv'
    folds = []
    for i in range(5):
      f = pd.read_csv(distance_cv_folder / (
        "preds_bag_fold_" + str(i) + ".csv"))
      f['fold'] = i
      folds.append(f)
    combined_train_folds = pd.concat(folds)
    
    valid_preds = pd.read_csv(distance_folder / 'predictions' / (
      "distance_valid.csv"))
    valid_preds.drop(valid_preds.columns[0], axis=1, inplace=True)
    valid_preds['fold'] = np.nan
    
    all_preds = pd.concat([combined_train_folds, valid_preds])
    all_preds.sort_values(
      ['site', 'fn', 'sub_trajectory_id'], inplace=True)
    all_preds.index = np.arange(all_preds.shape[0])
    device_errors['dist_pred'] = np.nan
    device_errors.loc[np.where(~np.isnan(device_errors.dist.values))[0],
                      'dist_pred'] = all_preds.pred.values
    device_errors['dist_error'] = device_errors.dist.values-(
      device_errors.dist_pred.values)
    device_errors['rel_dist_error'] = device_errors.dist_error.values/(
      device_errors.dist_pred.values)
    device_errors.sort_values(['fn', 'sub_trajectory_id'], inplace=True)
    device_errors.index = np.arange(device_errors.shape[0])
    device_errors['rel_weight'] = np.concatenate(
      device_errors.groupby('fn').apply(
        lambda x: np.abs(x.dist.values)/np.abs(x.dist.values).sum()))
    device_errors['section'] = "Middle"
    device_errors.loc[np.where(device_errors.sub_trajectory_id.values == (
      device_errors.num_waypoints.values-2))[0], 'section'] = "Last"
    device_errors.loc[np.where(
      device_errors.sub_trajectory_id.values == 0)[0], 'section'] = "First"
    device_errors['middle_weight_sums'] = np.concatenate(
      device_errors.groupby('fn').apply(
        lambda x: np.repeat((
          x.rel_weight.values[x.section.values=="Middle"]).sum(),
          x.shape[0]).reshape(-1)))
    device_errors.sort_values(
      ['device_id', 'first_last_wifi_time', 'sub_trajectory_id'], inplace=True)
    device_errors.index = np.arange(device_errors.shape[0])
    device_errors['rel_middle_weight'] = 0
    middle_rows = np.where(device_errors.section.values == "Middle")[0]
    device_errors.loc[middle_rows, "rel_middle_weight"] = (
      device_errors.rel_weight.values[middle_rows]/(
        device_errors.middle_weight_sums.values[middle_rows]))
    device_errors['angle_error'] = np.arctan2(
      device_errors.y_pred.values, device_errors.x_pred.values) - np.arctan2(
        device_errors.y.values, device_errors.x.values)
    change_rows = np.where((~np.isnan(device_errors.angle_error.values)) & (
      device_errors.angle_error.values < np.pi))[0]
    device_errors.loc[change_rows, 'angle_error'] = (
      device_errors.angle_error.values[change_rows] + 2*np.pi)
    change_rows = np.where((~np.isnan(device_errors.angle_error.values)) & (
      device_errors.angle_error.values > np.pi))[0]
    device_errors.loc[change_rows, 'angle_error'] = (
      device_errors.angle_error.values[change_rows] - 2*np.pi)
    
    def f(x):
      d = {}
      d['site'] = x['site'].values[0]
      d['floor'] = x['floor'].values[0]
      d['mode'] = x['mode'].values[0]
      d['train_fold'] = x['train_fold'].values[0]
      d['num_waypoints'] = x['num_waypoints'].values[0]
      d['total_dist'] = x['dist'].values.sum()
      
      d['mean_rel_dist_error'] = (x['rel_dist_error'].values*(
        x['rel_weight'].values)).sum()
      d['mean_abs_rel_dist_error'] = (np.abs(x['rel_dist_error'].values)*(
        x['rel_weight'].values)).sum()
      d['mean_angle_error'] = (x['angle_error'].values*(
        x['rel_weight'].values)).sum()
      d['mean_abs_angle_error'] = (np.abs(x['angle_error'].values)*(
        x['rel_weight'].values)).sum()
      d['first_rel_dist_error'] = x['rel_dist_error'].values[0]
      d['first_abs_rel_dist_error'] = np.abs(x['rel_dist_error'].values)[0]
      d['first_angle_error'] = x['angle_error'].values[0]
      d['first_abs_angle_error'] = np.abs(x['angle_error'].values)[0]
      d['middle_mean_rel_dist_error'] = (x['rel_dist_error'].values*(
        x['rel_weight'].values))[1:-1].sum()
      d['middle_mean_abs_rel_dist_error'] = (np.abs(x[
        'rel_dist_error'].values)*(x['rel_weight'].values))[1:-1].sum()
      d['middle_mean_angle_error'] = (x['angle_error'].values*(
        x['rel_weight'].values))[1:-1].sum()
      d['middle_mean_abs_angle_error'] = (np.abs(x['angle_error'].values)*(
        x['rel_weight'].values))[1:-1].sum()      
      d['last_rel_dist_error'] = x['rel_dist_error'].values[-1]
      d['last_abs_rel_dist_error'] = np.abs(x['rel_dist_error'].values)[-1]
      d['last_angle_error'] = x['angle_error'].values[-1]
      d['last_abs_angle_error'] = np.abs(x['angle_error'].values)[-1]
      
      d['first_first_last_wifi_time'] = (
        x['first_last_wifi_time'].values).min()
      d['time'] = (x['start_time'].values).min()
      d['device_id'] = (x['device_id'].values).min()
      
      return pd.Series(d, index=list(d.keys()))

    fn_dev_errors = device_errors.groupby('fn').apply(f).reset_index()
    
    fn_dev_errors['plot_time'] = fn_dev_errors['time']
    fn_dev_errors.loc[np.where(
      fn_dev_errors['mode'].values == "test")[0], 'plot_time'] = (
      fn_dev_errors.first_first_last_wifi_time.values[
        np.where(fn_dev_errors['mode'] == "test")[0]])
    fn_dev_errors['row'] = 1+np.arange(fn_dev_errors.shape[0])
    fn_dev_errors.sort_values(['device_id', 'plot_time'], inplace=True)
    fn_dev_errors.to_csv(save_path_fn_errors, index=False)