import datetime
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path

import utils

# Strategy 0: subtract 0.11 from the dense inner grid ratio, predict an inner grid when the max ratio >= 1.0666
# Strategy 1: Predict based on the best ratio without biasing or thresholding (threshold 1.0)
# Strategy 2: Average the inner grid predictions that have a ratio >= 1.0
def run(mode, strategy_id):
  print(f"Ensembling predictions for ensembling id {strategy_id}")
  inner_predict_threshold = 1.0666 if strategy_id == 0 else 1.0
  align_linear_non_grid = False
  aligned_angle_tolerance = np.pi/36
  save_results = False
  save_test = True
  save_test_extended = True
  print_changed_private = False
  
  valid_mode = mode == 'valid'
  if valid_mode:
    baseline_grid_source = 'valid - walls_only_old.csv' # V3, 'min_distance_to_known': 3.0, 'max_distance_to_known': 30.0,
    other_sources = [
      'valid - sparse_inner.csv',  # V4, 'min_distance_to_known': 3.0, 'max_distance_to_known': 30.0, generate_inner_waypoints True, generate_edge_waypoints False 
      'valid - dense_inner.csv',  # V4, 'min_distance_to_known': 1.5, 'max_distance_to_known': 30.0, generate_inner_waypoints True, generate_edge_waypoints False - wall_point_distance_multiplier 0.2, inner_point_distance_multiplier 0.35
      ]
  else:
    baseline_grid_source = 'test - walls_only_old - extended.csv' # V3, 'min_distance_to_known': 3.0, 'max_distance_to_known': 30.0,
    other_sources = [
      'test - sparse_inner - extended.csv',  # V4, 'min_distance_to_known': 3.0, 'max_distance_to_known': 30.0, generate_inner_waypoints True, generate_edge_waypoints False 
      'test - dense_inner - extended.csv',  # V4, 'min_distance_to_known': 1.5, 'max_distance_to_known': 30.0, generate_inner_waypoints True, generate_edge_waypoints False - wall_point_distance_multiplier 0.2, inner_point_distance_multiplier 0.35
      ]
  other_penalty_regularization = 0.11*int(strategy_id == 0)*np.arange(
    len(other_sources))
    
  data_folder = utils.get_data_folder()
  storage_folder = Path(data_folder).parent / 'Combined predictions'
  
  save_ensemble_test_folder = data_folder / 'final_submissions'
  pathlib.Path(save_ensemble_test_folder).mkdir(parents=True, exist_ok=True)
  test_save_path = save_ensemble_test_folder / (
    'final_submissions_ensemble_' + str(strategy_id))
  if mode == 'test' and test_save_path.is_file():
    return  
  
  # Load the raw data 
  baseline = pd.read_csv(storage_folder / baseline_grid_source)
  other = [pd.read_csv(storage_folder / s) for s in other_sources]
    
  def angles_close(a1, a2, tolerance):
    abs_angle_diff = np.abs(a1 - a2)
    angle_diff = min(np.abs(abs_angle_diff), 2*np.pi - abs_angle_diff)
    
    return angle_diff <= tolerance
  
  baseline_fn = baseline.fn.values
  baseline_preds = baseline[['x_pred', 'y_pred']].values
  other_fns = [o.fn.values for o in other]
  other_preds = [o[['x_pred', 'y_pred']].values for o in other]
  fns = np.sort(np.unique(baseline_fn))
  results = []
  merged_predictions = np.copy(baseline_preds)
  for fn in fns:
    baseline_ids = np.where(baseline_fn == fn)[0]
    other_ids = [np.where(f == fn)[0] for f in other_fns]
    
    baseline_penalty = baseline.selected_total_penalty.values[
      baseline_ids].sum()
    other_penalties = np.array([
      o.selected_total_penalty.values[id_].sum() for (o, id_) in zip(
        other, other_ids)])
    
    preds_equal = np.array([
      np.all(np.isclose(baseline_preds[baseline_ids], p[id_])) for (
        p, id_) in zip(other_preds, other_ids)])
    
    penalty_ratios = baseline_penalty/other_penalties
    biased_penalty_ratios = penalty_ratios - other_penalty_regularization
    replace_scores = biased_penalty_ratios-100*(preds_equal.astype(np.int32))
    override_individual = replace_scores > inner_predict_threshold
    override = np.any(override_individual)
    replace_id = None
    if override:
      replace_id = np.argmax(replace_scores)
      if strategy_id == 2:
        non_baseline_ids = np.where(override_individual)[0]
        override_preds = np.zeros_like(merged_predictions[baseline_ids])
        for id_ in non_baseline_ids:
          override_preds += other_preds[id_][other_ids[id_]]
        assert override_preds.sum() != 0
        merged_predictions[baseline_ids] = (override_preds/non_baseline_ids.size)
      else:
        merged_predictions[baseline_ids] = other_preds[replace_id][
          other_ids[replace_id]]
      align_non_grid_extended = other[replace_id].iloc[other_ids[replace_id]]
      
      considered_override_ids = np.where(
        replace_scores > inner_predict_threshold)[0]
      overrides_equal = considered_override_ids.size > 1
      for o_id in considered_override_ids:
        if o_id != replace_id:
          if not np.all(np.isclose(merged_predictions[baseline_ids],
                                   other_preds[o_id][other_ids[o_id]])):
            overrides_equal = False
            break
    else:
      align_non_grid_extended = baseline.iloc[baseline_ids]
      overrides_equal = True
      
    addit_grid = align_non_grid_extended.waypoint_train_count_pred.values == 0
    tweak_segments = []
    if align_linear_non_grid and np.any(addit_grid):
      # Identify additional grid waypoints that fall on a > 1 segment line and
      # can be tweaked using the sensor data
      fn_preds = align_non_grid_extended[['x_pred', 'y_pred']].values
      segment_angles = np.angle(np.diff(fn_preds[:, 0] + 1j*fn_preds[:, 1]))
      num_waypoints = addit_grid.size
      start = None
      active = False
      active_angle = None
      for i in range(1, num_waypoints):
        if addit_grid[i]:
          if not active:
            start = i-1
            active = True
            active_angle = segment_angles[i-1]
          else:
            this_aligned = angles_close(
              active_angle, segment_angles[i-1], aligned_angle_tolerance)
            if this_aligned:
              active_angle = np.angle(fn_preds[i, 0] - fn_preds[start, 0] + 1j*(
                fn_preds[i, 1] - fn_preds[start, 1]))
            else:
              end = i-1
              if (end - start) > 1:
                tweak_segments.append((start, end))
              start = i-1
              active = True
              active_angle = segment_angles[i-1]
        else:
          if active:
            # End an active group here, save it if the aligned length > 2
            total_angle = np.angle(fn_preds[i, 0] - fn_preds[start, 0] + 1j*(
              fn_preds[i, 1] - fn_preds[start, 1]))
            this_aligned = angles_close(
              total_angle, segment_angles[i-1], aligned_angle_tolerance)
            end = int(this_aligned) + i - 1
            
            # if fn == 'a0c05d232b01247a9ba48be0':
            #   import pdb; pdb.set_trace()
            #   x=1
            
            if (end - start) > 1:
              tweak_segments.append((start, end))
            start = None
            active = False
            active_angle = None
            
      for s, e in tweak_segments:
        segment_start = fn_preds[s]
        segment_end = fn_preds[e]
        segment_vector = segment_end-segment_start
        dist_preds = align_non_grid_extended.segment_dist_pred.values[(
          s+1):(e+1)]
        cum_fractions = np.cumsum(dist_preds)/dist_preds.sum()
        optim_pos = segment_start.reshape((1, -1)) + segment_vector.reshape(
          (1, -1))*((cum_fractions[:-1]).reshape((-1, 1)))
        override_rows = baseline_ids[(s+1):e]
        merged_predictions[override_rows] = optim_pos
    
    fn_summary = {
      'fn': fn,
      
      'preds_equal': preds_equal,
      'num_waypoints': align_non_grid_extended.shape[0],
      'override': override,
      'replace_id': replace_id,
      'overrides_equal': overrides_equal,
      'override_last': override and replace_id == (len(other_sources)-1),
      'tweak_segments': tweak_segments,
      
      'baseline_penalty': baseline_penalty,
      'other_penalties': other_penalties,
      }
    for i in range(penalty_ratios.size):
      fn_summary['override_' + str(i)] = override_individual[i]
      fn_summary['penalty_ratios_' + str(i)] = penalty_ratios[i]
    
    if mode == 'test':
      fn_summary['leaderboard_type'] = other[0].leaderboard_type.values[
        other_ids[0][0]]
    
    results.append(fn_summary)
    
  results_df = pd.DataFrame(results)
  
  if save_results:
    record_time = str(datetime.datetime.now())[:19]
    results_path = storage_folder / (record_time + ' - multiple grid combo.csv')
    results_df.to_csv(results_path, index=False)
  
  if mode == "valid":
    original_error = baseline.after_optim_error.values.mean()
    after_merge_error = np.sqrt(
      (baseline.x_actual.values - merged_predictions[:, 0])**2 + (
        baseline.y_actual.values - merged_predictions[:, 1])**2).mean()
    print(f"Original error: {original_error}; after merge error: {after_merge_error}")
  else:
    # Analyze the impact on the public and private set
    public_diff_results_df = results_df[(
      results_df.leaderboard_type.values == 'public') & ~(
        np.stack(results_df.preds_equal.values)[:, 0]) & (
          results_df.override.values)]  
    private_diff_results_df = results_df[(
      results_df.leaderboard_type.values == 'private') & ~(
        np.stack(results_df.preds_equal.values)[:, 0]) & (
          results_df.override.values)]
        
    if print_changed_private:
      with pd.option_context('mode.chained_assignment', None):
        private_diff_results_df.sort_values(
          ['penalty_ratios_0'], ascending=False, inplace=True)
      ratios = list(zip(private_diff_results_df.fn.values,
                        private_diff_results_df.penalty_ratios_0.values))
      print(f"Changed private: {ratios}")
          
    if save_test:
      short_baseline_path = storage_folder / (
        baseline_grid_source[:-15] + '.csv')
      submission = pd.read_csv(short_baseline_path)
      
      match_rows = []
      prev_fn = None
      for i, sps in enumerate(submission.site_path_timestamp):
        fn = sps.split('_')[1]
        if i == 0 or fn != prev_fn:
          match_rows.append(np.where(fn == baseline_fn)[0])
        prev_fn = fn
      match_rows = np.concatenate(match_rows)
      
      submission.loc[:, 'x'] = merged_predictions[match_rows, 0]
      submission.loc[:, 'y'] = merged_predictions[match_rows, 1]
      
      save_ext = 'Blend: ' + baseline_grid_source[:-4]
      for s in other_sources:
        save_ext += ('; ' + s[:-4])
      align_ext = " - Align" if align_linear_non_grid else ""
      av_inner_ext = " - Average inner" if strategy_id == 2 else ""
      save_ext += av_inner_ext + align_ext + " - Threshold " + str(
        inner_predict_threshold)
      save_path = storage_folder / (save_ext + '.csv')
      submission.to_csv(save_path, index=False)
      submission.to_csv(test_save_path, index=False)
      
      if save_test_extended:
        submission_extended = baseline.copy()
        submission_extended.loc[:, 'x_pred'] = merged_predictions[:, 0]
        submission_extended.loc[:, 'y_pred'] = merged_predictions[:, 1]
        
        extended_save_ext = save_ext + ' - extended'
        extended_save_path = storage_folder / (extended_save_ext + '.csv')
        submission_extended.to_csv(extended_save_path, index=False)