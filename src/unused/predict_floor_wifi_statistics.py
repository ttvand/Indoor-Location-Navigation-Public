import numpy as np
import pandas as pd

import utils


data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
preds_folder = data_folder.parent / 'Models' / 'non_parametric_wifi' / (
    'predictions')
save_path = preds_folder / ('combined_top_wifi_floor_preds.csv')
test_floor_comparison = pd.read_csv(data_folder / 'test_floor_comparison.csv')
leaderboard_types = pd.read_csv(data_folder / 'leaderboard_type.csv')
all_combined = []

for mode in ['valid', 'test']:
  load_path = preds_folder / (mode + '_floor_pred_distances_v2.csv')
  data = pd.read_csv(load_path)
  
  fns = np.sort(np.unique(data.fn))
  outputs = []
  for fn in fns:
    fn_data = data[data.fn == fn]
    
    floors = fn_data.floor.values
    numeric_floors = fn_data.numeric_floor.values
    mean_min_distances = fn_data.mean_min_distance.values
    sort_ids = np.argsort(mean_min_distances)
    min_min_distances = fn_data.min_min_distance.values
    sort_min_ids = np.argsort(min_min_distances)
    
    min_dist = mean_min_distances[sort_ids[0]]
    second_dist = mean_min_distances[sort_ids[1]]
    third_dist = mean_min_distances[sort_ids[2]]
    min_min_min_dist = min_min_distances[sort_min_ids[0]]
    
    if mode == 'valid':
      test_mode = None
    else:
      test_mode = leaderboard_types['type'].values[
        np.where(fn == leaderboard_types.fn.values)[0][0]]
    
    append_vals = {
      'fn': fn,
      'mode': mode,
      'test_mode': test_mode,
      
      'min_dist_floor': floors[sort_ids[0]],
      'min_dist_numeric_floor': numeric_floors[sort_ids[0]],
      'min_mean_min_distance': min_dist,
      
      'second_distance_gap': second_dist-min_dist,
      'second_mean_min_distance': second_dist,
      'second_dist_floor': floors[sort_ids[1]],
      'second_dist_numeric_floor': numeric_floors[sort_ids[1]],
      
      'third_distance_gap': third_dist-min_dist,
      'third_mean_min_distance': third_dist,
      'third_dist_floor': floors[sort_ids[2]],
      'third_dist_numeric_floor': numeric_floors[sort_ids[2]],
      
      'min_min_min_distance': min_min_min_dist,
      'min_min_dist_floor': floors[sort_min_ids[0]],
      'min_min_dist_numeric_floor': numeric_floors[sort_min_ids[0]],
      }
    
    if mode == 'valid':
      append_vals['actual_floor'] = fn_data.reference_floor_label.values[0]
    
    outputs.append(append_vals)
    
  combined = pd.DataFrame(outputs)
  all_combined.append(combined)
  
  if mode == 'test':
    test_floor_comparison_sorted = test_floor_comparison.sort_values(['fn'])
    test_floor_comparison_sorted.index = np.arange(
      test_floor_comparison_sorted.shape[0])
    assert np.all(test_floor_comparison_sorted.fn.values == combined.fn.values)
    disagree_ids = np.where(test_floor_comparison_sorted.tom_pred.values != (
      combined.min_dist_numeric_floor.values))[0]
    print(f"Disagree ids: {disagree_ids}")
    
    disagree_min_ids = np.where(
      test_floor_comparison_sorted.tom_pred.values != (
        combined.min_min_dist_numeric_floor.values))[0]
    print(f"Min disagree ids: {disagree_min_ids}")
    
  else:
    valid_accuracy = (combined.min_dist_numeric_floor.values == (
      combined.actual_floor.values)).mean()
    print(f"Validation accuracy: {valid_accuracy}")
    
all_combined_df = pd.concat(all_combined)
all_combined_df.to_csv(save_path, index=False)