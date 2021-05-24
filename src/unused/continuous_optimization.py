from functools import partial
import numpy as np
import pandas as pd
import pathlib
import scipy
from scipy.optimize import minimize
import scipy.sparse.linalg

debug_fn = [None, '5da84747ae6cfc0006ca8268'][0]
mode = ['cost_optimization_public', 'cost_optimization_custom'][1]

repo_data_folder = pathlib.Path(__file__).parent.absolute().parent / (
  'Data files')

wifi_source_nn = repo_data_folder / 'valid - 2021-04-23 11:52:52.csv'
wifi_source_lgbm = repo_data_folder / 'valid_predictions_lgbm_v2.csv'
sensor_distance_source = repo_data_folder / 'dsp' / 'distance_valid.csv'
sensor_relative_movement_source = repo_data_folder / 'dsp' / (
  'relative_movement_v2_valid.csv')
sensor_abs_movement_source = repo_data_folder / 'dsp' / (
  'relative_movement_v3_valid.csv')

wifi_lgbm = pd.read_csv(wifi_source_lgbm)
wifi_nn = pd.read_csv(wifi_source_nn)
sensor_distance = pd.read_csv(sensor_distance_source)
sensor_relative_movement = pd.read_csv(sensor_relative_movement_source)
sensor_abs_movement = pd.read_csv(sensor_abs_movement_source)

def optim_preds_spsolve(fn_wifi, xy_hat, delta_xy_hat):
  T_ref = fn_wifi.waypoint_time.values
  N = xy_hat.shape[0]
  delta_t = np.diff(T_ref)
  alpha = (8.1)**(-2) * np.ones(N)
  beta = 5*((0.3 + 0.3 * 1e-3 * delta_t[1:])**(-2))
  
  A = scipy.sparse.spdiags(alpha, [0], N, N)
  B = scipy.sparse.spdiags(beta, [0], N-1, N-1)
  D = scipy.sparse.spdiags(np.stack(
    [-np.ones(N), np.ones(N)]), [0, 1], N-1, N)
  Q = A + (D.T @ B @ D)
  c = (A @ xy_hat) + (D.T @ (B @ delta_xy_hat))

  xy_star = scipy.sparse.linalg.spsolve(Q, c)
  
  return xy_star

def trajectory_loss(positions, wifi_positions, pred_distances, delta_xy_hat):
  distance_pen_constant = 0*25
  distance_laplace_smoother = 1
  relative_distance_pen_constant = 10
  
  # Compute the penalty for consistency with the wifi predictions
  positions = positions.reshape((-1, 2))
  wifi_error = np.sqrt(((positions-wifi_positions)**2).sum(1)).mean()
  
  # Penalize for the distance between predictions
  pos_changes = positions[1:] - positions[:-1]
  pos_distances = np.sqrt((pos_changes**2).sum(1))
  distance_error = distance_pen_constant*(np.abs(1-(
      pos_distances+distance_laplace_smoother)/(
        pred_distances+distance_laplace_smoother))).sum()
        
  # Penalize for the relative distance between predictions
  pos_change_errors = pos_changes-delta_xy_hat
  relative_distance_error = relative_distance_pen_constant*np.abs(
    pos_change_errors).sum()
  
  loss = wifi_error + 0*distance_error + relative_distance_error
  
  return loss

def optim_preds_spoptimize(wifi_positions, fn_distance, delta_xy_hat):
  pred_distances = fn_distance.pred.values
  
  loss = partial(
    trajectory_loss, wifi_positions=wifi_positions,
    pred_distances=pred_distances, delta_xy_hat=delta_xy_hat)
  x0 = wifi_positions.flatten().tolist()
  res = minimize(
      loss, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
  
  xy_star = res.x.reshape((-1, 2))
  
  return xy_star

all_fn = np.sort(np.unique(wifi_lgbm.fn.values))
results = []
for fn in all_fn:
  if debug_fn is None or fn == debug_fn:
    fn_wifi_lgbm = wifi_lgbm[wifi_lgbm.fn == fn]
    fn_wifi_nn = wifi_nn[wifi_nn.fn == fn]
    fn_distance = sensor_distance[sensor_distance.fn == fn]
    fn_relative_movement = sensor_relative_movement[
      sensor_relative_movement.fn == fn]
    fn_abs_movement = sensor_abs_movement[sensor_abs_movement.fn == fn]
    
    actual_xy = fn_wifi_lgbm[['x', 'y']].values
    xy_hat_lgbm = fn_wifi_lgbm[['x_preds', 'y_preds']].values
    xy_hat_nn = fn_wifi_nn[
      ['x_before_optim_pred', 'y_before_optim_pred']].values
    xy_hat = 0.75*xy_hat_nn+0.25*xy_hat_lgbm
    delta_xy_hat = fn_abs_movement[['x_pred', 'y_pred']].values
    if mode == 'cost_optimization_public':
      xy_star = optim_preds_spsolve(fn_wifi_lgbm, xy_hat, delta_xy_hat)
    else:
      xy_star = optim_preds_spoptimize(xy_hat, fn_distance, delta_xy_hat)
    
    for i in range(actual_xy.shape[0]):
      results.append({
        'fn': fn,
        'x': actual_xy[i, 0],
        'y': actual_xy[i, 1],
        'before_optim_x': xy_hat[i, 0],
        'before_optim_y': xy_hat[i, 1],
        'before_optim_error': np.sqrt(((xy_hat[i]-actual_xy[i])**2).sum()),
        'after_optim_x': xy_star[i, 0],
        'after_optim_y': xy_star[i, 1],
        'after_optim_error': np.sqrt(((xy_star[i]-actual_xy[i])**2).sum()),
        })

    before_error = np.sqrt(((xy_hat-actual_xy)**2).mean(1)).mean()
    after_error = np.sqrt(((xy_star-actual_xy)**2).mean(1)).mean()
    after_change = after_error-before_error
    print(f"{fn} {before_error:.2f} ({after_change:.2f})")

results = pd.DataFrame(results)

if debug_fn is None:
  before_overall_error = results.before_optim_error.values.mean()
  after_overall_error = results.after_optim_error.values.mean()
  after_change = after_overall_error - before_overall_error
  print()
  print(f"Overall: {before_overall_error:.2f}({after_change:.2f})")
  print(f"Overall stats: {before_overall_error:.2f}\
({after_overall_error:.2f})")
