import lightgbm as lgb
import numpy as np
import pandas as pd
import pathlib
import pickle

import utils

def run(mode):
  print("Model the sensor uncertainty")
  fn_mode = ['mean', 'joined_middle_last', 'first_middle_last'][0]
  skip_unbias_models = not True
  overwrite_models = True
  max_train_folds = [1, None][1]
  additional_feature_cols = ['num_waypoints']
  
  params = {
      'objective': 'regression',
      'learning_rate': 0.005,
      'extra_trees': True,
      'num_leaves': 40,
      'n_estimators': int(1e3),
      'max_depth': -1,
      'min_child_samples': 1,
      'colsample_bynode': 0.4,
      'subsample_freq': 1,
      'subsample': 0.8,
      'metric': 'rmse',
      'verbose': -1,
      'n_jobs': 1
  }
  
  
  data_folder = utils.get_data_folder()
  model_folder = data_folder.parent / 'Models' / 'correct_sensor_preds'
  save_ext  = '' if fn_mode == 'mean' else ' ' + fn_mode
  predict_ext = mode + save_ext + '.csv'
  save_folder = model_folder / 'predictions'
  pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
  save_path = save_folder / predict_ext
  if save_path.is_file():
    return
  load_ext = '' if fn_mode == 'mean' else ' first_middle_last'
  data_path = model_folder / (mode + load_ext + '.csv')
  data = pd.read_csv(data_path)
  target_cols = [c for c in data.columns if c[-7:] =='_target']
  last_non_feature_col = target_cols[-1]
  
  if fn_mode == 'joined_middle_last':
    target_cols = [c[6:] for c in target_cols if c[:5] == 'first']
  
  last_non_feature_id = np.where(data.columns == last_non_feature_col)[0][0]
  feature_cols = additional_feature_cols + data.columns.tolist()[
    (last_non_feature_id+1):]
  non_feature_cols = data.columns.tolist()[:(last_non_feature_id+1)]
  
  def prepare_data(data, mode, fn_mode, target_cols, feature_cols=None):
    sub = data[(data['mode'] == mode)]
    
    if fn_mode == 'joined_middle_last':
      nrow = sub.shape[0]
      orig_sub = sub.copy()
      sub = pd.concat([sub, sub, sub])
      sub.index = np.arange(sub.shape[0])
      for c in target_cols:
        sub[c] = np.concatenate([
          orig_sub['first_' + c].values,
          orig_sub['middle_mean_' + c].values,
          orig_sub['last_' + c].values,
          ])
      sub['segment_type'] = np.repeat(np.arange(3), nrow)
      
      if feature_cols is not None:
        feature_cols.append('segment_type')
    
    return sub
  
  folds = data.train_fold.values[data['mode'] == 'train'].astype(np.int32)
  unique_folds = np.sort(np.unique(folds))
  if max_train_folds is not None:
    unique_folds = unique_folds[:max_train_folds]
  num_folds = unique_folds.size
  predict_rows = np.where(data['mode'].values == mode)[0]
  predict_data = prepare_data(data, mode, fn_mode, target_cols, feature_cols)
  predict_features = predict_data[feature_cols].values
  combined = {k: data[k].values[predict_rows] for k in non_feature_cols}
  
  for target_col_id, target_col in enumerate(target_cols):
    unbias_target = not 'abs_' in target_col
    if not unbias_target or not skip_unbias_models:
      print(f'\nTarget {target_col_id+1} of {len(target_cols)}: {target_col}')
      predict_targets = predict_data[target_col].values
      predict_fold_preds = {}
      predict_fold_preds_l = []
      for f_id, f in enumerate(unique_folds):
        print(f"Fold {f_id+1} of {num_folds}")
        model_path = model_folder / (
          target_col + ' - fold ' + str(int(f)) + save_ext + '.pickle')
        if mode == 'valid' and (not model_path.is_file() or overwrite_models):
          train_data = prepare_data(data, 'train', fn_mode, target_cols)
          train_data = train_data[train_data['train_fold'] != f]
          train_features = train_data[feature_cols].values
          train_targets = train_data[target_col].values
          non_nan_train_targets = ~np.isnan(train_targets)
          
          model = lgb.LGBMRegressor(**params)
          model = model.fit(train_features[non_nan_train_targets],
                            train_targets[non_nan_train_targets], verbose=1)
          with open(model_path, "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
          with open(model_path, "rb") as file:
            model = pickle.load(file)
            
        preds = model.predict(predict_features)
        predict_fold_preds[target_col + '_fold_' + str(f)] = preds
        predict_fold_preds_l.append(preds)
        
      avg_fold_pred = np.stack(predict_fold_preds_l, -1).mean(-1)
      predict_fold_preds[target_col + '_avg_fold'] = avg_fold_pred
      
      if mode == 'valid':
        if unbias_target:
          original_avg_abs_error = np.abs(predict_targets).mean()
          debiased_avg_abs_error = np.abs(
            predict_targets-avg_fold_pred).mean()
          print(f"Orig abs error: {original_avg_abs_error:.3f};\
     Debiased abs error: {debiased_avg_abs_error:.3f}")
        else:
          abs_err_correlation = np.corrcoef(np.stack(
            [predict_targets, avg_fold_pred]))[0, 1]
          print(f"Abs error correlation: {abs_err_correlation:.3f}")
        
      if fn_mode == 'joined_middle_last':
        orig_keys = list(predict_fold_preds.keys())
        for k in orig_keys:
          for st_id, st in enumerate(['first_', 'middle_mean_', 'last_']):
            start_index = st_id*predict_rows.size
            end_index = (st_id+1)*predict_rows.size
            predict_fold_preds[st + k] = predict_fold_preds[k][
              start_index:end_index]
        for k in orig_keys:
          del predict_fold_preds[k]
        
      combined.update(predict_fold_preds)
    
  combined_df = pd.DataFrame(combined)
  combined_df.to_csv(save_path, index=False)