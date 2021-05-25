import numpy as np
import pandas as pd

import utils

def run(mode):
  print("Calculating the sensor uncertainty for unseen trajectories")
  save_uncertainties = True
  distance_model_dist_estimates = True
  apply_angle_correction_threshold = 0 # >= 0.4 means no corrections
  apply_distance_correction_threshold = 0 # >= 0.4 means no corrections
  predictions_ext = mode + '.csv'
  
  source_valid_test_predictions = [
    'relative_movement_v3_valid.csv', 'relative_movement_v3_test.csv']
  source_valid_test_dist_predictions = [
    'distance_valid.csv', 'distance_test.csv']
  fn_mode = 'first_middle_last' if 'first_middle_last' in predictions_ext else (
    'joined_middle_last' if 'joined_middle_last' in predictions_ext else 'mean')
  
  data_folder = utils.get_data_folder()
  model_folder = data_folder.parent / 'Models' / 'correct_sensor_preds'
  uncertainty_path = model_folder / 'predictions' / (
      'uncertainty - ' + predictions_ext)
  if uncertainty_path.is_file():
    return
  
  source_pred_folder = data_folder.parent / 'Models' / (
    'sensor_absolute_movement') / 'predictions'
  source_ext = source_valid_test_predictions[int(mode == 'test')]
  source_path = source_pred_folder / source_ext
  preds = pd.read_csv(source_path)
  
  source_dist_pred_folder = data_folder.parent / 'Models' / (
    'sensor_distance') / 'predictions'
  source_dist_ext = source_valid_test_dist_predictions[int(mode == 'test')]
  source_dist_path = source_dist_pred_folder / source_dist_ext
  dist_preds = pd.read_csv(source_dist_path)
  
  corrections_path = model_folder / 'predictions' / predictions_ext
  corrections = pd.read_csv(corrections_path)
  uncertainties = pd.read_csv(source_path)
  
  
  fns = np.sort(np.unique(preds.fn.values))
  preds['distance'] = np.sqrt(preds['x']**2 + preds['y']**2)
  if distance_model_dist_estimates:
    assert np.all(dist_preds.fn.values == preds.fn.values)
    preds['pred_distance'] = dist_preds.pred.values
  else:
    preds['pred_distance'] = np.sqrt(preds['x_pred']**2 + preds['y_pred']**2)
  preds['pred_distance_corrected'] = preds['pred_distance']
  preds['x_pred_corrected'] = preds['x_pred']
  preds['y_pred_corrected'] = preds['y_pred']
  uncertainties['distance_uncertainty'] = np.nan
  uncertainties['pred_distance_uncertainty'] = np.nan
  uncertainties['angle_uncertainty'] = np.nan
  uncertainties['pred_angle_uncertainty'] = np.nan
  
  uncertainty_map_mean = [
    ('mean_abs_rel_dist_error_target', 'distance_uncertainty'),
    ('mean_abs_rel_dist_error_target_avg_fold', 'pred_distance_uncertainty'),
    ('mean_abs_angle_error_target', 'angle_uncertainty'),
    ('mean_abs_angle_error_target_avg_fold', 'pred_angle_uncertainty'),
    ]
  uncertainty_map_first = [
    ('first_abs_rel_dist_error_target', 'distance_uncertainty'),
    ('first_abs_rel_dist_error_target_avg_fold', 'pred_distance_uncertainty'),
    ('first_abs_angle_error_target', 'angle_uncertainty'),
    ('first_abs_angle_error_target_avg_fold', 'pred_angle_uncertainty'),
    ]
  uncertainty_map_middle = [
    ('middle_mean_abs_rel_dist_error_target', 'distance_uncertainty'),
    ('middle_mean_abs_rel_dist_error_target_avg_fold',
     'pred_distance_uncertainty'),
    ('middle_mean_abs_angle_error_target', 'angle_uncertainty'),
    ('middle_mean_abs_angle_error_target_avg_fold', 'pred_angle_uncertainty'),
    ]
  uncertainty_map_last = [
    ('last_abs_rel_dist_error_target', 'distance_uncertainty'),
    ('last_abs_rel_dist_error_target_avg_fold', 'pred_distance_uncertainty'),
    ('last_abs_angle_error_target', 'angle_uncertainty'),
    ('last_abs_angle_error_target_avg_fold', 'pred_angle_uncertainty'),
    ]
  
  for fn in fns:
    pred_rows = np.where(preds.fn.values == fn)[0]
    uncertainty_rows = np.where(uncertainties.fn.values == fn)[0]
    correct_row = np.where(corrections.fn.values == fn)[0][0]
    
    if fn_mode == 'mean':
      if 'mean_rel_dist_error_target_avg_fold' in corrections.columns:
        mean_correction = corrections[
          'mean_rel_dist_error_target_avg_fold'].values[correct_row]
        if np.abs(mean_correction) > apply_distance_correction_threshold:
          preds.loc[pred_rows, 'pred_distance_corrected'] = preds[
            'pred_distance'].values[pred_rows]*(1+mean_correction)
        
      if 'mean_angle_error_target_avg_fold' in corrections.columns:
        angle_correction = corrections[
          'mean_angle_error_target_avg_fold'].values[correct_row]
        if np.abs(angle_correction) > apply_angle_correction_threshold:
          x = preds.x_pred.values[pred_rows]
          y = preds.y_pred.values[pred_rows]
          preds.loc[pred_rows, 'x_pred_corrected'] = np.cos(
            angle_correction)*x + np.sin(angle_correction)*y
          preds.loc[pred_rows, 'y_pred_corrected'] = -np.sin(
            angle_correction)*x + np.cos(angle_correction)*y
        
      for k1, k2 in uncertainty_map_mean:
        v = corrections[k1].values[correct_row]
        uncertainties.loc[uncertainty_rows, k2] = v
    else:
      first_correction = corrections[
        'first_rel_dist_error_target_avg_fold'].values[correct_row]
      middle_correction = corrections[
        'middle_mean_rel_dist_error_target_avg_fold'].values[correct_row]
      last_correction = corrections[
        'last_rel_dist_error_target_avg_fold'].values[correct_row]
      
      if np.abs(first_correction) > apply_distance_correction_threshold:
        preds.loc[pred_rows[0], 'pred_distance_corrected'] = preds[
          'pred_distance'].values[pred_rows[0]]*(1+first_correction)
      if np.abs(middle_correction) > apply_distance_correction_threshold:
        preds.loc[pred_rows[1:-1], 'pred_distance_corrected'] = preds[
          'pred_distance'].values[pred_rows[1:-1]]*(1+middle_correction)
      if np.abs(last_correction) > apply_distance_correction_threshold:
        preds.loc[pred_rows[-1], 'pred_distance_corrected'] = preds[
          'pred_distance'].values[pred_rows[-1]]*(1+last_correction)
        
      for k1, k2 in uncertainty_map_first:
        v = corrections[k1].values[correct_row]
        uncertainties.loc[uncertainty_rows[0], k2] = v
        
      for k1, k2 in uncertainty_map_middle:
        v = corrections[k1].values[correct_row]
        uncertainties.loc[uncertainty_rows[1:-1], k2] = v
        
      for k1, k2 in uncertainty_map_last:
        v = corrections[k1].values[correct_row]
        uncertainties.loc[uncertainty_rows[-1], k2] = v
      
  preds['pred_error'] = np.abs((preds['distance'] - preds['pred_distance']))
  preds['corrected_error'] = np.abs(
    (preds['distance'] - preds['pred_distance_corrected']))
  preds['rel_pred_dist_error'] = (preds['distance'] - preds['pred_distance'])/(
    preds['pred_distance'])
  preds['rel_corrected_dist_error'] = (preds['distance'] - preds[
    'pred_distance_corrected'])/(preds['pred_distance_corrected'])
  orig_abs_err = (np.abs(preds['x'] - preds['x_pred']) + np.abs(
    preds['y'] - preds['y_pred'])).mean()/2
  corrected_abs_err = (np.abs(preds['x'] - preds['x_pred_corrected']) + np.abs(
    preds['y'] - preds['y_pred_corrected'])).mean()/2
  
  mean_orig_rel_dist_err = np.abs(preds['rel_pred_dist_error'].values).mean()
  mean_corrected_rel_dist_err = np.abs(
    preds['rel_corrected_dist_error'].values).mean()
  rel_err = mean_corrected_rel_dist_err/mean_orig_rel_dist_err
  print(f'Corrected relative error rate: {rel_err:.3f}')
  
  orig_dist_mae = preds.pred_error.values.mean()
  corrected_dist_mae = preds.corrected_error.values.mean()
  print(f'Original MAE: {orig_dist_mae:.3f}; Corrected MAE:\
   {corrected_dist_mae:.3f}')
  print(f'Rel move original MAE: {orig_abs_err:.3f}; Corrected rel move MAE:\
   {corrected_abs_err:.3f}')
   
  changed_fraction = (preds['pred_error'] != preds['corrected_error']).mean()
  improved_fraction = (preds['pred_error'] > preds['corrected_error']).mean()/(
    (changed_fraction+1e-9))
  print(f'Improved distance pred fraction: {improved_fraction:.3f}')
  
  dist_uncertainty_cor = np.corrcoef(np.stack([
    uncertainties['pred_distance_uncertainty'].values,
    uncertainties['distance_uncertainty'].values,
    ]))[0, 1]
  angle_uncertainty_cor = np.corrcoef(np.stack([
    uncertainties['pred_angle_uncertainty'].values,
    uncertainties['angle_uncertainty'].values,
    ]))[0, 1]
  print(f'Distance uncertainty correlation: {dist_uncertainty_cor:.3f}')
  print(f'Angle uncertainty correlation: {angle_uncertainty_cor:.3f}')
  uncertainties.plot.scatter('pred_distance_uncertainty', 'distance_uncertainty')
  
  uncertainties = uncertainties[[
    'site', 'floor', 'fn', 'sub_trajectory_id', 'num_waypoints',
    'distance_uncertainty', 'pred_distance_uncertainty', 'angle_uncertainty',
    'pred_angle_uncertainty',
    ]]
  if save_uncertainties:
    uncertainty_path = model_folder / 'predictions' / (
      'uncertainty - ' + predictions_ext)
    uncertainties.to_csv(uncertainty_path, index=False)