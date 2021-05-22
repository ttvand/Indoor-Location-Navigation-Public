import numpy as np
import pandas as pd
import pickle

import utils

def run():
  data_folder = utils.get_data_folder()
  summary_path = data_folder / 'file_summary.csv'
  df = pd.read_csv(summary_path)
  model_folder = data_folder.parent / 'Models' / 'sensor_absolute_movement'
  fold_folder = model_folder / 'errors cv'
  valid_path = model_folder / 'predictions' / 'relative_movement_v3_valid.csv'
  device_id_path = data_folder / 'device_ids.pickle'
  meta_sensor_path = data_folder / 'sensor_data' / 'meta.csv'
  
  with open(device_id_path, 'rb') as f:
    device_ids = pickle.load(f)
  meta_data = pd.read_csv(meta_sensor_path, dtype={'test_type': object})
  
  
  #####################
  # A: XXX
  #####################
  
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
  
  
  #####################
  # B: XXX
  #####################
  
  import pdb; pdb.set_trace()
  train_preds_list = []
  for i in range(5):
    preds = pd.read_csv(fold_folder / f'preds_bag_fold_{i}.csv')
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
    
  save_path = data_folder / 'sensor_model_device_errors.csv'
  all_preds.sort_values([
    'device_id', 'first_last_wifi_time', 'sub_trajectory_id'], inplace=True)
  all_preds.to_csv(save_path, index=False)