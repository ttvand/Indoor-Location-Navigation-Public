import numpy as np
import pandas as pd
import utils

def run():
  print("Creating validation set")
  require_valid_waypoints_in_train = True
  max_valid_unique_fraction = 0.4
  max_valid_unique_count_per_trajectory = 15
  prob_allow_hardest_trajectories = 0.15
  min_waypoints_holdout = 6  # Reflect the test data - only put long trajectories in the holdout set
  holdout_fraction = 0.08
  np.random.seed(14)
  data_folder = utils.get_data_folder()
  summary_path = data_folder / 'file_summary.csv'
  stratified_holdout_path = data_folder / 'holdout_ids.csv'
  if not stratified_holdout_path.is_file():
    df = pd.read_csv(summary_path)
    
    if require_valid_waypoints_in_train:
      train_waypoints, waypoint_counts = utils.get_train_waypoints(
        data_folder, df)
    
    sites = sorted(set(df.site_id))
    valid_trajectory_seen_train = []
    floor_dfs = []
    for s in sites:
      floors = sorted(
          set(df.text_level[(df.site_id == s).values
                            & (df['mode'] == 'train').values]))
      for f in floors:
        floor_df = df.iloc[(df.site_id == s).values & (
          df.text_level == f).values]
        if require_valid_waypoints_in_train:
          try:
            floor_int = utils.TEST_FLOOR_MAPPING[f]
          except:
            floor_int = utils.NON_TEST_FLOOR_MAPPING[f]
          floor_waypoints = train_waypoints.iloc[
              (train_waypoints.site_id.values == s)
              & (train_waypoints.level.values.astype(np.float32) == floor_int)]
          floor_waypoint_counts = {(str(k[2]), str(k[3])): v
                                   for k, v in waypoint_counts.items()
                                   if k[0] == s and (k[1] == floor_int)}
    
          considered_seq_ids = []
          fn_sorted_ids = np.argsort(-floor_df.num_train_waypoints.values)
          for i, fn in enumerate(floor_df.fn.values[fn_sorted_ids]):
            fn_waypoints = floor_waypoints[floor_waypoints.fn == fn]
            if fn_waypoints.shape[0] == 0:
              continue
            if fn_waypoints.shape[0] < min_waypoints_holdout and len(
                considered_seq_ids) > 0:
              break
            waypoint_counts_fn = np.array([
                floor_waypoint_counts[(str(x), str(y))]
                for x, y in zip(fn_waypoints.x.values, fn_waypoints.y.values)
            ])
            this_waypoint_counts = np.zeros(fn_waypoints.shape[0])
            waypoint_vals = np.stack(
              [fn_waypoints.x.values, fn_waypoints.y.values])
            for j in range(fn_waypoints.shape[0]):
              this_waypoint_counts[j] = (
                  (waypoint_vals[0] == waypoint_vals[0, j]) &
                  (waypoint_vals[1] == waypoint_vals[1, j])).sum()
    
            unseen_normalized = (waypoint_counts_fn > this_waypoint_counts) / (
                this_waypoint_counts)
            non_unique_count = unseen_normalized.sum()
            total_count = (1 / this_waypoint_counts).sum()
            unique_fraction = 1 - (non_unique_count / total_count)
            unique_count = np.round(total_count - non_unique_count)
    
            # if (unique_fraction <= max_valid_unique_fraction) and not (
            #     unique_count <= max_valid_unique_count_per_trajectory):
            #   import pdb; pdb.set_trace()
            #   x=1
    
            if (unique_fraction <= max_valid_unique_fraction) and (
                unique_count <= max_valid_unique_count_per_trajectory) or (
                  np.random.uniform() < prob_allow_hardest_trajectories):
              considered_seq_ids.append(fn_sorted_ids[i])
              for x, y in zip(fn_waypoints.x.values, fn_waypoints.y.values):
                floor_waypoint_counts[(str(x), str(y))] -= 1
    
          if len(considered_seq_ids) == 0:
            if floor_df.test_site.values[0]:
              raise ValueError("No valid validation trajectories selected")
            considered_seq_ids = [fn_sorted_ids[0]]
    
          considered_seq_ids = np.array(considered_seq_ids)
        else:
          considered_seq_ids = np.where(
            floor_df.num_train_waypoints.values >= (min(
              min_waypoints_holdout, floor_df.num_train_waypoints.max())))[0]
        # if (s, f) == ('5d2709a003f801723c3251bf', '3F'):
        #   import pdb; pdb.set_trace()
        num_holdout_trajectories = min(
            considered_seq_ids.size,
            max(1, int(floor_df.shape[0] * holdout_fraction)))
        if num_holdout_trajectories == considered_seq_ids.size:
          holdout_ids = considered_seq_ids
        else:
          probs = floor_df.num_train_waypoints.values[considered_seq_ids]**1
          probs = probs / probs.sum()
          holdout_ids = np.random.choice(
              considered_seq_ids, num_holdout_trajectories, replace=False,
              p=probs)
        floor_waypoint_counts = {
          (str(k[2]), str(k[3])): v for k, v in waypoint_counts.items() if k[
            0] == s and (k[1] == floor_int)}
        for fn in floor_df.fn.values[holdout_ids]:
          fn_waypoints = floor_waypoints[floor_waypoints.fn == fn]
          for x, y in zip(fn_waypoints.x.values, fn_waypoints.y.values):
            floor_waypoint_counts[(str(x), str(y))] -= 1
        
        num_train_waypoints = floor_df.num_train_waypoints.values
        test_site = floor_df.test_site.values
        floor_df = floor_df.iloc[:, :4]
        floor_df.index = np.arange(floor_df.shape[0])
        floor_df['text_level'] = f
        floor_df['holdout'] = False
        floor_df.loc[holdout_ids, 'holdout'] = True
        floor_df.loc[holdout_ids, 'mode'] = 'valid'
        floor_df['num_train_waypoints'] = num_train_waypoints
        floor_df['test_site'] = test_site
    
        for valid_fn in floor_df.fn.values[floor_df.holdout.values]:
          fn_waypoints = floor_waypoints[
            floor_waypoints.fn == valid_fn][['x', 'y']].values
          if df.test_site.values[(df.site_id == s)][0]:
            waypoints_in_train = np.array([
              floor_waypoint_counts[
                (str(fn_waypoints[i, 0]),
                 str(fn_waypoints[i, 1]))] > 0 for i in (
                   range(fn_waypoints.shape[0]))])
            valid_trajectory_seen_train.append((
              waypoints_in_train.shape[0], waypoints_in_train.sum()))
    
        # if s == '5d2709c303f801723c3299ee' and f == '1F':
        #   import pdb; pdb.set_trace()
        #   x=1
    
        floor_dfs.append(floor_df)
    
    valid_trajectory_seen_train = np.array(valid_trajectory_seen_train)
    entire_traj_in_train = (
      valid_trajectory_seen_train[:, 0] == valid_trajectory_seen_train[:, 1])
    entire_waypoints_fraction = valid_trajectory_seen_train[
      entire_traj_in_train, 0].sum()/valid_trajectory_seen_train[:, 0].sum()
    print(entire_traj_in_train.mean(), entire_waypoints_fraction.mean(),
          valid_trajectory_seen_train[:, 1].sum()/valid_trajectory_seen_train[
            :, 0].sum())
    
    combined_df = pd.concat(floor_dfs)
    combined_df.to_csv(stratified_holdout_path, index=False)
