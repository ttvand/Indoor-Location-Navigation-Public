import numpy as np
import pathlib
import pickle

import pandas as pd
import utils

def run():
  only_process_test_sites = True
  sensor_cols = ['time', 'acce', 'gyro', 'ahrs']
  
  data_folder = utils.get_data_folder()
  save_folder = data_folder / 'sensor_data'
  pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
  summary_path = data_folder / 'file_summary.csv'
  source_submission = 'submission_cost_minimization.csv'
  submission_folder = data_folder / 'submissions'
  submission = pd.read_csv(submission_folder / source_submission)
  submission = utils.override_test_floor_errors(submission)
  sample_sub_fns = np.array([sps.split('_')[1] for sps in (
    submission.site_path_timestamp)])
  sample_sub_times = np.array([int(sps.split('_')[2]) for sps in (
    submission.site_path_timestamp)])
  holdout_df = pd.read_csv(data_folder / 'holdout_ids.csv')
  df = pd.read_csv(summary_path)
  
  # Overwrite the validation data mode
  validation_fns = set(holdout_df.fn.values[holdout_df.holdout])
  df['mode'] = [
      'valid' if fn in validation_fns else m
      for (fn, m) in zip(df['fn'].values, df['mode'].values)
  ]
  
  
  def rotate_by_angles(orig, angles):
    rotated = np.zeros_like(orig)
  
    for i, theta in enumerate(angles):
      o = orig[i]
      c, s = np.cos(theta), np.sin(theta)
      r = np.matrix([[c, s], [-s, c]])
      rotated[i] = np.dot(r, o)
  
    return rotated
  
  
  # Combine all the between waypoint sub-trajectories into a single file for each
  # data mode
  target_sensor_cols = None
  for mode in np.unique(df['mode'].values):
    # mode = 'valid'
    waypoint_mapping = {}
  
    sub_df = df.iloc[np.where(df['mode'] == mode)[0]]
  
    if only_process_test_sites:
      sub_df = sub_df[sub_df.test_site.values]
      save_ext = ''
    else:
      save_ext = '_all_sites'
    save_path = save_folder / (mode + save_ext + '.pickle')
    
    if save_path.is_file():
      continue
    print(mode)
  
    for fn_id, (fn, site, floor) in enumerate(
        zip(sub_df.fn, sub_df.site_id, sub_df.text_level)):
      #print(fn_id)
      path_ext = fn + '_reshaped.pickle'
      if mode == 'test':
        data_path = data_folder / mode / path_ext
        sub_fn_ids = np.where(sample_sub_fns == fn)[0]
        waypoint_times = sample_sub_times[sub_fn_ids]
        floor_int = submission.floor.values[sub_fn_ids[0]]
        waypoints = None
        relative_waypoint_movement_1 = None
        relative_waypoint_distances = None
        relative_waypoint_movement_2 = None
      else:
        try:
          floor_int = utils.TEST_FLOOR_MAPPING[floor]
        except:
          print(f"Failed {fn_id}")
          continue
        data_path = data_folder / 'train' / site / floor / path_ext
  
      try:
        with open(data_path, 'rb') as f:
          file_data = pickle.load(f)
      except:
        print(f"Failed {fn_id}")
        continue
  
      if mode != 'test':
        waypoint_times = file_data['waypoint'].time.values
        waypoints = file_data['waypoint']
        waypoint_pos = waypoints[['x_waypoint', 'y_waypoint']].values
        relative_waypoint_movement_1 = np.diff(waypoint_pos, axis=0)
        rel_angles = np.angle(relative_waypoint_movement_1[:, 0] + 1j *
                              (relative_waypoint_movement_1[:, 1]))
        relative_waypoint_movement_2 = rotate_by_angles(
            relative_waypoint_movement_1[1:], rel_angles[:-1])
        relative_waypoint_distances = np.sqrt(
            (relative_waypoint_movement_1**2).sum(1))
  
      num_waypoints = waypoint_times.size
  
      # Chunk out the waypoint segments
      waypoint_segments = []
      fractions_time_covered = []
      shared_time = file_data['shared_time']
      share_time_vals = shared_time.time.values
      for i in range(num_waypoints - 1):
        start_time = waypoint_times[i]
        end_time = waypoint_times[i + 1]
  
        if target_sensor_cols is None:
          target_sensor_cols = [
              c for c in shared_time.columns
              if any([sc in c for sc in sensor_cols])
          ]
  
        start_row = max(0, (share_time_vals <= start_time).sum() - 1)
        end_row = min(share_time_vals.size,
                      (share_time_vals < end_time).sum() + 1)
        
        time_range = end_time - start_time
        covered_time = min(end_time, share_time_vals[end_row-1]) - max(
          start_time, share_time_vals[start_row])
        fractions_time_covered.append(covered_time/time_range)
  
        # if fn == '5dc8e91a17ffdd0006f12ce0' and i == 0:
        #   import pdb; pdb.set_trace()
        #   x=1
  
        waypoint_segments.append(shared_time.iloc[np.arange(start_row, end_row)])
  
      # if fn == '5dc8e91a17ffdd0006f12ce0':
      #   import pdb; pdb.set_trace()
      #   x=1
  
      # import pdb; pdb.set_trace()
      waypoint_mapping[fn] = {
          'site': site,
          'floor': floor_int,
          'num_waypoints': num_waypoints,
          'waypoints': waypoints,
          'waypoint_times': waypoint_times,
          'fractions_time_covered': np.array(fractions_time_covered),
          'waypoint_segments': waypoint_segments,
          'relative_waypoint_movement_1': relative_waypoint_movement_1,
          'relative_waypoint_distances': relative_waypoint_distances,
          'relative_waypoint_movement_2': relative_waypoint_movement_2,
      }
  
    # Save the combined mapping to disk
    with open(save_path, 'wb') as handle:
      pickle.dump(waypoint_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)