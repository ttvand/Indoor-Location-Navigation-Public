import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import utils

only_process_test_sites = True

data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
train_waypoints_path = data_folder / 'train_waypoints.csv'
submission_folder = data_folder / 'submissions'
submission_path = submission_folder / 'probe_max_submission.csv'
df = pd.read_csv(summary_path)

# Loop over all file paths and keep track of the max train floor, X and Y for
# all waypoints
waypoint_counts = {}
waypoint_floor = []
waypoint_x = []
waypoint_y = []
all_waypoints = []
for i in range([100, df.shape[0]][1]):
  print(i)
  if (df['mode'][i] == 'train') and (not only_process_test_sites or
                                     df.test_site[i]):
    pickle_path = data_folder / (
        str(Path(df.ext_path[i]).with_suffix('')) + '_reshaped.pickle')
    with open(pickle_path, 'rb') as f:
      trajectory = pickle.load(f)

    site = df.site_id[i]
    floor = df.level[i]
    for w_id in range(trajectory['waypoint'].shape[0]):
      x = trajectory['waypoint']['x_waypoint'][w_id]
      y = trajectory['waypoint']['y_waypoint'][w_id]
      k = (site, floor, x, y)
      if k in waypoint_counts:
        waypoint_counts[k] += 1
      else:
        waypoint_counts[k] = 1

      waypoint_floor.append(floor)
      waypoint_x.append(x)
      waypoint_y.append(y)

      fn = Path(df['ext_path'][i]).stem
      all_waypoints.append(tuple([fn] + list(k)))

max_floor = np.array(waypoint_floor).max()
max_x = np.array(waypoint_x).max()
max_y = np.array(waypoint_y).max()
print(f"Max floor: {max_floor}")
print(f"Max x: {max_x}")
print(f"Max y: {max_y}")
train_waypoints = pd.DataFrame(np.array(all_waypoints))
train_waypoints.columns = ["fn", "site_id", "level", "x", "y"]
train_waypoints.to_csv(train_waypoints_path, index=False)

sample_submission = pd.read_csv(data_folder / 'sample_submission.csv')

# Generate the probing default dummy submission
sample_submission['floor'] = max_floor + 1
sample_submission['x'] = max_x + 1
sample_submission['y'] = max_y + 1
Path(submission_folder).mkdir(parents=True, exist_ok=True)
sample_submission.to_csv(submission_path, index=False)
