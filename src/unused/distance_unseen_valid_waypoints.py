import numpy as np
import pandas as pd
import pickle

from hallways.calc_old import generate_waypoints_3
from hallways.calc import generate_waypoints_4
import utils

try_load_waypoints_from_disk = True
save_waypoints_to_disk = True
# CONSIDERED_SITES = ["5da1382d4db8ce0c98bbe92e"]
CONSIDERED_SITES = None

CONSIDERED_FLOORS = None
# CONSIDERED_FLOORS = {"F1", "1F"}
# CONSIDERED_FLOORS = {"B1", "F1", "1F"}
# CONSIDERED_FLOORS = {"B1"}
# CONSIDERED_FLOORS = {"F2", "2F"}


grid_version = [3, 4][1]
config = {
  'min_distance_to_known': 3.0,
  'max_distance_to_known': 30.0,
  'generate_inner_waypoints': False,
  'generate_edge_waypoints': True,
  'wall_point_distance_multiplier': 0.4,
  'inner_point_distance_multiplier': 0.7,
  }

data_folder = utils.get_data_folder()

df = pd.read_csv(data_folder / 'file_summary.csv')
waypoints_path = data_folder / 'train_waypoints_timed.csv'
waypoints = pd.read_csv(waypoints_path)
waypoints_folder = data_folder / 'stashed_floor_additional_waypoints'
site_floors = df.iloc[df.test_site.values].groupby(
    ['site_id', 'level', 'text_level']).size().reset_index()
site_floors.rename(
    columns={site_floors.columns[3]: "total_num_fn"}, inplace=True)
site_floors['num_train_waypoints'] = np.nan
site_floors['num_valid_waypoints'] = np.nan
site_floors['num_valid_unseen_waypoints'] = np.nan
site_floors['num_unique_train_waypoints'] = np.nan
site_floors['num_additional_grid_waypoints'] = np.nan
site_floors['valid_unseen_waypoints_mean_distance'] = np.nan
site_floors['valid_unseen_waypoints_max_distance'] = np.nan
site_floors['valid_unseen_waypoints_sum_distance'] = np.nan
site_floors['additional_waypoint_ratio'] = np.nan

num_floors = site_floors.shape[0]
for i in range(num_floors):
  site = site_floors.site_id.values[i]
  floor = site_floors.text_level.values[i]

  if CONSIDERED_SITES is not None and site not in CONSIDERED_SITES:
    continue
  if CONSIDERED_FLOORS is not None and floor not in CONSIDERED_FLOORS:
    continue

  print(f'Processing floor {i+1} of {num_floors}')

  all_floor_waypoint_ids = np.where((
    waypoints.site_id.values == site) & (
      waypoints.text_level.values == floor))[0]
  train_floor_waypoint_ids = all_floor_waypoint_ids[
    waypoints['mode'].values[all_floor_waypoint_ids] == 'train']
  valid_floor_waypoint_ids = all_floor_waypoint_ids[
    waypoints['mode'].values[all_floor_waypoint_ids] == 'valid']
  num_train_waypoints = train_floor_waypoint_ids.size
  num_valid_waypoints = valid_floor_waypoint_ids.size
  site_floors.loc[i, 'num_train_waypoints'] = num_train_waypoints
  site_floors.loc[i, 'num_valid_waypoints'] = num_valid_waypoints

  train_locations = waypoints.iloc[train_floor_waypoint_ids][
    ['x_waypoint', 'y_waypoint']].values
  known_waypoints = np.unique(train_locations, axis=0)
  num_unique_train_waypoints = known_waypoints.shape[0]
  site_floors.loc[i, 'num_unique_train_waypoints'] = num_unique_train_waypoints
  waypoints_path = waypoints_folder / (
      'valid_' + site + '_' + str(floor) + '_' + str(grid_version) + '_' +
      str(float(config['min_distance_to_known'])) + '_' +
      str(float(config['max_distance_to_known'])) + '_' +
      str(config['generate_inner_waypoints']) + '_' +
      str(config['generate_edge_waypoints']) + '_' +
      str(float(config['wall_point_distance_multiplier'])) + '_' +
      str(float(config['inner_point_distance_multiplier'])) + '.pickle')
  if try_load_waypoints_from_disk:
    should_generate_waypoints = not waypoints_path.is_file()
  else:
    should_generate_waypoints = True
  if should_generate_waypoints:
    grid_f = generate_waypoints_3 if grid_version == 3 else (
      generate_waypoints_4)
    wall_waypoints, inner_waypoints = grid_f(
        site=site,
        floor=floor,
        known_waypoints=known_waypoints,
        min_distance_to_known=config['min_distance_to_known'],
        max_distance_to_known=config['max_distance_to_known'],
        generate_inner_waypoints=config['generate_inner_waypoints'],
        generate_edge_waypoints=config['generate_edge_waypoints'],
        wall_point_distance_multiplier=config[
          'wall_point_distance_multiplier'],
        inner_point_distance_multiplier=config[
          'wall_point_distance_multiplier'],
        )
    generated_waypoints = np.concatenate((wall_waypoints, inner_waypoints))

    if save_waypoints_to_disk:
      with open(waypoints_path, 'wb') as handle:
        pickle.dump(
          generated_waypoints, handle, protocol=pickle.HIGHEST_PROTOCOL)
  else:
    # The path exists!
    with open(waypoints_path, 'rb') as f:
      wall_waypoints, inner_waypoints = pickle.load(f)
    generated_waypoints = np.concatenate((wall_waypoints, inner_waypoints))

  site_floors.loc[i, 'num_additional_grid_waypoints'] = (
    generated_waypoints.shape[0])

  combined_waypoints = np.concatenate(
    [known_waypoints, generated_waypoints], 0)

  valid_locations = waypoints.iloc[valid_floor_waypoint_ids][
    ['x_waypoint', 'y_waypoint']].values
  x_diff = combined_waypoints[:, :1] - np.expand_dims(valid_locations[:, 0], 0)
  y_diff = combined_waypoints[:, 1:] - np.expand_dims(valid_locations[:, 1], 0)
  valid_distances = np.sqrt(x_diff**2 + y_diff**2)
  valid_unseen_locations = (
    valid_distances[:num_unique_train_waypoints].min(0)) > 0
  site_floors.loc[i, 'num_valid_unseen_waypoints'] = (
    valid_unseen_locations.sum())
  valid_unseen_nearest_distances = valid_distances[
    :, valid_unseen_locations].min(0)

  if valid_unseen_nearest_distances.size == 0:
    valid_unseen_nearest_distances = np.array([np.nan])

  site_floors.loc[i, 'valid_unseen_waypoints_mean_distance'] = (
    valid_unseen_nearest_distances.mean())
  site_floors.loc[i, 'valid_unseen_waypoints_max_distance'] = (
    valid_unseen_nearest_distances.max())
  site_floors.loc[i, 'valid_unseen_waypoints_sum_distance'] = (
    valid_unseen_nearest_distances.sum())
  site_floors.loc[i, 'additional_waypoint_ratio'] = (
    generated_waypoints.shape[0])/known_waypoints.shape[0]

sum_new_dist = np.nansum(
  site_floors.valid_unseen_waypoints_sum_distance.values)
train_grid_count = site_floors.num_unique_train_waypoints.values.sum()
new_grid_count = np.nansum(site_floors.num_additional_grid_waypoints.values)
print(f'Total summed distance new waypoints: {sum_new_dist:.1f}')
print(f'Number of train grid waypoints: {train_grid_count:.0f}')
print(f'Number of additional grid waypoints: {new_grid_count:.0f}')
