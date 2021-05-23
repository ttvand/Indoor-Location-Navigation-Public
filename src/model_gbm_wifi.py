import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from pathlib import Path
from src import utils
from joblib import Parallel, delayed
import math
import lightgbm as lgb

OUT_PATH = 'data/processed/gbm_wifi'
T_DELTA = 8000
NO_SIGNAL = -200

def interpolate_wp(df, n=1):
  outdf = df.copy().sort_values('time')
  outdf['interpolated'] = False
  if len(outdf) < 2:
    return outdf
  new_recs = []
  for i in range(len(outdf) - 1):
    for j in range(n):
      w = (j + 1) / (n + 1)
      new_recs.append({
          'time': (outdf.iloc[i]['time'] + outdf.iloc[i + 1]['time']) // 2,
          'floor':
              outdf.iloc[i]['floor'],
          'x_waypoint':
              w * outdf.iloc[i]['x_waypoint'] +
              (1 - w) * outdf.iloc[i + 1]['x_waypoint'],
          'y_waypoint':
              w * outdf.iloc[i]['y_waypoint'] +
              (1 - w) * outdf.iloc[i + 1]['y_waypoint'],
          'interpolated':
              True
      })
  outdf = pd.concat([outdf, pd.DataFrame(new_recs)]).sort_values('time')
  return outdf

class GBMds():

  def __init__(self, bssid_list, beacon_list):
    self.base_path = utils.get_data_folder()
    self.bssid_dict = {v: i for i, v in enumerate(bssid_list)}
    self.ibeacon_dict = {v: i for i, v in enumerate(beacon_list)}
    self.sub = pd.read_csv(f"{self.base_path}/sample_submission.csv")
    self.sub['fn'] = self.sub['site_path_timestamp'].apply(
        lambda z: z.split('_')[1])
    self.sub['time'] = self.sub['site_path_timestamp'].apply(
        lambda z: int(z.split('_')[2]))
    self.sub = self.sub[['fn', 'time', 'x',
                         'y']].rename({
                             'x': 'x_waypoint',
                             'y': 'y_waypoint'
                         },
                                      axis=1)

  def read_one_rec(self, rec):
    data_path = self.base_path / (
        str(Path(rec['ext_path']).with_suffix('')) + '_reshaped.pickle')
    with open(data_path, 'rb') as f:
      file_data = pickle.load(f)
      rec2 = {**rec, **file_data}
    return rec2

  def process_row(self, row, ds_type):
    out = []
    rec = self.read_one_rec(row)

    if ds_type == 'test':
      rec['waypoint'] = self.sub[self.sub['fn'] == rec['fn']].copy()

    if True:  # second timestamp correction
      temp = rec['wifi'].groupby('t1_wifi')['t2_wifi'].max().reset_index()
      corr = (temp['t2_wifi'] - temp['t1_wifi']).max() + 324
      rec['shared_time']['time'] += corr
      rec['wifi']['t1_wifi'] += corr
      rec['waypoint']['time_orig'] = rec['waypoint']['time']
      rec['waypoint']['time'] += corr
      if 'ibeacon' in rec:
        rec['ibeacon']['t1_beac'] += corr

    rec['wifi']['bssid_wifi'] = rec['wifi']['bssid_wifi'].map(self.bssid_dict)
    if 'ibeacon' in rec:
      rec['ibeacon']['id_beac_3'] = rec['ibeacon']['id_beac_3'].map(
          self.ibeacon_dict)

    for _, r in rec['waypoint'].iterrows():
      out_delta = {
          **{f"var_wifi_{i}": NO_SIGNAL for i in range(len(self.bssid_dict))},
          **{
              f"var_beacon_{i}": NO_SIGNAL
              for i in range(len(self.ibeacon_dict))
          },
          'x': r['x_waypoint'],
          'y': r['y_waypoint'],
          'waypoint_time': r['time_orig'],
          'site_id': rec['site_id'],
          'level': rec['level'],
          'text_level': rec['text_level'],
          'fn': rec['fn'],
      }

      for j, v in rec['wifi'][(rec['wifi']['t2_wifi'] - r['time']).abs(
      ) < T_DELTA].groupby('bssid_wifi')['rssid_wifi'].max().iteritems():
        out_delta[f"var_wifi_{int(j)}"] = v
      if 'ibeacon' in rec:
        for j, v in rec['ibeacon'][(rec['ibeacon']['t1_beac'] - r['time']).abs(
        ) < T_DELTA].groupby('id_beac_3')['rssi_beac'].max().iteritems():
          out_delta[f"var_beacon_{int(j)}"] = v
      magn = rec['shared_time'][(rec['shared_time']['time'] -
                                 r['time']).abs() < T_DELTA][[
                                     'x_magn', 'y_magn', 'z_magn'
                                 ]].mean()
      out_delta = {
          'var_x_magn': magn['x_magn'],
          'var_y_magn': magn['y_magn'],
          'var_z_magn': magn['z_magn'],
          **out_delta
      }
      out.append(out_delta)

    return pd.DataFrame(out)

  def process_df(self, df, ds_type):
    #out = Parallel(n_jobs=-1)(delayed(self.process_row)(row)
    #                          for _, row in tqdm(df.iterrows(), total=len(df)))
    #out = Parallel(n_jobs=-1)(
    #    delayed(self.process_row)(row) for _, row in df.iterrows())
    out = [self.process_row(row, ds_type) for _, row in df.iterrows()]
    return pd.concat([x for x in out if x is not None])

def process_location(ds_type, ds, df0, site, level):
  df_processed = ds.process_df(
      df0[(df0['site_id'] == site)
          & (df0['level'] == level)], ds_type)
  if ds_type == "train":
    cols = [
        col for col in df_processed.columns
        if df_processed[col].nunique() > 1 or (not col.startswith('var_'))
    ]
  else:
    cols = pd.read_csv(f"{OUT_PATH}/train/{site}_{level:1.1f}.csv").columns
  df_processed = df_processed[cols]
  df_processed.to_csv(
      f"{OUT_PATH}/{ds_type}/{site}_{level:1.1f}.csv", index=False)

def prepare_data():

    with open('data/processed/tests_bssid_wifi.p', 'rb') as handle:
        bssid_wifi = pickle.load(handle)
    with open('data/processed/tests_id_beac_3.p', 'rb') as handle:
        id_beac_3 = pickle.load(handle)

    df = pd.read_csv('data/file_summary.csv')
    df_test = df[df['mode'] == 'test'].copy().reset_index(drop=True)
    sub = pd.read_csv('data/submission_cost_minimization.csv')
    sub['id'] = sub['site_path_timestamp'].apply(
        lambda z: z.split('_')[0] + '_' + z.split('_')[1])
    sub = utils.override_test_floor_errors(sub)
    sub = sub.groupby('id').first()['floor'].copy()

    df_test['level'] = df_test.apply(
        lambda z: z['site_id'] + '_' + z['fn'], axis=1).map(sub)

    df = df[(~df['num_train_waypoints'].isnull()) & (df['num_wifi'] > 0) &
            (df['test_site'])]

    ds = GBMds(bssid_wifi, id_beac_3)

    os.makedirs(f"{OUT_PATH}/train/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}/test/", exist_ok=True)

    locations = df.groupby(['site_id', 'level']).size().index
    Parallel(n_jobs=-1)(delayed(process_location)("train", ds, df, site, level)
                        for site, level in tqdm(locations))

    locations = df_test.groupby(['site_id', 'level']).size().index
    Parallel(n_jobs=-1)(
        delayed(process_location)("test", ds, df_test, site, level)
        for site, level in tqdm(locations))

    print("GBM data preparation done")


def rotate_origin_only(x, y, radians):
  xx = x * math.cos(radians) + y * math.sin(radians)
  yy = -x * math.sin(radians) + y * math.cos(radians)
  return xx, yy


def fit_single_model(fl, holdout_fns, params, fullfit=False):
  df = pd.read_csv(f"{OUT_PATH}/train/{fl}")

  if fullfit:
    train_df = df
    valid_df = pd.read_csv(f"{OUT_PATH}/test/{fl}")
  else:
    is_valid = df['fn'].isin(holdout_fns)
    train_df = df[~is_valid]
    valid_df = df[is_valid].copy()

  if 'interpolated' in valid_df.columns:
    valid_df = valid_df[~valid_df['interpolated']]
    del valid_df['interpolated']

  X_train = train_df[[c for c in train_df.columns if c.startswith('var_')
                     ]].values
  X_valid = valid_df[[c for c in valid_df.columns if c.startswith('var_')
                     ]].values

  model = lgb.LGBMRegressor(**params)
  model = model.fit(X_train, train_df['x'].values, verbose=0)
  valid_df['x_preds'] = model.predict(X_valid)

  model = lgb.LGBMRegressor(**params)
  model = model.fit(X_train, train_df['y'].values, verbose=0)
  valid_df['y_preds'] = model.predict(X_valid)

  x_tr, y_tr = rotate_origin_only(train_df['x'].values, train_df['y'].values,
                                    np.pi / 4)
  x_vl, y_vl = rotate_origin_only(valid_df['x'].values, valid_df['y'].values,
                                    np.pi / 4)
  model = lgb.LGBMRegressor(**params)
  model = model.fit(X_train, x_tr, verbose=0)
  xx = model.predict(X_valid)

  model = lgb.LGBMRegressor(**params)
  model = model.fit(X_train, y_tr, verbose=0)
  yy = model.predict(X_valid)

  xx, yy = rotate_origin_only(xx, yy, -np.pi / 4)

  valid_df['x_preds'] = (valid_df['x_preds'] + xx) / 2
  valid_df['y_preds'] = (valid_df['y_preds'] + yy) / 2

  return valid_df[[
      'x', 'y', 'site_id', 'level', 'text_level', 'fn', 'x_preds', 'y_preds',
      'waypoint_time'
  ]]

def fit(mode):
  params = {
    'objective': 'regression',
    'learning_rate': 0.005,
    'extra_trees': True,
    'num_leaves': 40,
    'n_estimators': 3_000,
    'max_depth': -1,
    'min_child_samples': 1,
    'colsample_bynode': 0.4,
    'subsample_freq': 1,
    'subsample': 0.8,
    'metric': 'rmse',
    'verbose': -1,
    'n_jobs': 1
  }

  holdout_ids = pd.read_csv('data/holdout_ids.csv')
  holdout_ids = holdout_ids[(holdout_ids['test_site'])
                          & (holdout_ids['mode'] == 'valid')]
  holdout_fns = set(holdout_ids['fn'])

  if mode=="valid":
    res = Parallel(n_jobs=-1)(
        delayed(fit_single_model)(fl, holdout_fns, params)
        for fl in tqdm(os.listdir(f"{OUT_PATH}/train")))

    res = pd.concat(res)
    res.to_csv("data/gbm_wifi_valid_predictions.csv", index=False)
    print(res.shape)
    print(
        np.sqrt((res['x'].values - res['x_preds'].values)**2 +
                (res['y'].values - res['y_preds'].values)**2).mean())
  
  if mode=="test":
    res = Parallel(n_jobs=-1)(
        delayed(fit_single_model)(fl, holdout_fns, params, fullfit=True)
        for fl in tqdm(os.listdir(f"{OUT_PATH}/test")))

    res = pd.concat(res)
    res.to_csv("data/gbm_wifi_test_predictions.csv", index=False)
    print(res.shape)
    print('GBM fitting finished')

