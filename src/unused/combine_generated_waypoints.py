import numpy as np
import pandas as pd
import pickle

from hallways.calc_old import generate_waypoints_3
from hallways.calc import generate_waypoints_4
import utils


mode = ['valid', 'test'][1]
store_generated_waypoints = True
try_load_waypoints_from_disk = True

grid_version = [3, 4][1]
grid_mode = ["standard", "dense", "very_dense"][2]
grid_settings = {
  "standard": {
    "min_distance_to_known": 3.0,
    "wall_point_distance_multiplier": 0.4,
    "inner_point_distance_multiplier": 0.7,
    },
  "dense": {
    "min_distance_to_known": 1.5,
    "wall_point_distance_multiplier": 0.2,
    "inner_point_distance_multiplier": 0.35,
    },
  "very_dense": {
    "min_distance_to_known": 1.125,
    "wall_point_distance_multiplier": 0.15,
    "inner_point_distance_multiplier": 0.262,
    },
  }
config = {
  'min_distance_to_known': grid_settings[grid_mode]["min_distance_to_known"],
  'max_distance_to_known': 30.0,
  'generate_inner_waypoints': True,
  'generate_edge_waypoints': False,
  'wall_point_distance_multiplier': grid_settings[grid_mode][
    "wall_point_distance_multiplier"],
  'inner_point_distance_multiplier': grid_settings[grid_mode][
    "inner_point_distance_multiplier"],
  }

data_folder = utils.get_data_folder()

df = pd.read_csv(data_folder / 'file_summary.csv')
waypoints_path = data_folder / 'train_waypoints_timed.csv'
waypoints = pd.read_csv(waypoints_path)
if mode == 'valid':
  waypoints = waypoints[waypoints['mode'] == 'train']
waypoints_folder = data_folder / 'stashed_floor_additional_waypoints'
site_floors = df.iloc[df.test_site.values].groupby(
  ['site_id', 'level', 'text_level']).size().reset_index()
all_generated_waypoints = []

num_floors = site_floors.shape[0]
for i in range(num_floors):
  site = site_floors.site_id.values[i]
  floor = site_floors.text_level.values[i]

  print(f'Processing floor {i+1} of {num_floors}')

  floor_waypoint_ids = np.where((
    waypoints.site_id.values == site) & (
      waypoints.text_level.values == floor))[0]
  
  train_locations = waypoints.iloc[floor_waypoint_ids][
    ['x_waypoint', 'y_waypoint']].values
  known_waypoints = np.unique(train_locations, axis=0)
  num_known = known_waypoints.shape[0]
  
  if try_load_waypoints_from_disk:
    waypoints_path = waypoints_folder / (
      mode + '_' + site + '_' + str(floor) + '_' + str(
        grid_version) + '_' + str(
          float(config['min_distance_to_known'])) + '_' + str(
            float(config['max_distance_to_known'])) + '_' + str(
              config['generate_inner_waypoints']) + '_' + str(
                config['generate_edge_waypoints']) + '_' + str(
                  float(config[
                    'wall_point_distance_multiplier'])) + '_' + str(
                      float(config[
                        'inner_point_distance_multiplier'])) +'.pickle')
    should_generate_waypoints = not waypoints_path.is_file()
  else:
    should_generate_waypoints = True
  if should_generate_waypoints:
    grid_f = generate_waypoints_3 if grid_version == 3 else (
      generate_waypoints_4)
    generated_waypoints_types = grid_f(
      site=site, floor=floor, known_waypoints=known_waypoints,
      min_distance_to_known=config['min_distance_to_known'],
      max_distance_to_known=config['max_distance_to_known'],
      generate_inner_waypoints=config['generate_inner_waypoints'],
      generate_edge_waypoints=config['generate_edge_waypoints'],
      wall_point_distance_multiplier=config[
        'wall_point_distance_multiplier'],
      inner_point_distance_multiplier=config[
        'wall_point_distance_multiplier'],
      )
  else:
    # The path exists!
    with open(waypoints_path, 'rb') as f:
      generated_waypoints_types = pickle.load(f)
    
  generated_waypoints = np.concatenate(generated_waypoints_types)
  combined_waypoints = np.concatenate([
    known_waypoints, generated_waypoints], 0)
  waypoint_types = ['train' if i < num_known else 'generated' for i in range(
    combined_waypoints.shape[0])]  
  
  all_generated_waypoints.append(pd.DataFrame({
    'site': site,
    'floor': floor,
    'type': waypoint_types,
    'x': combined_waypoints[:, 0],
    'y': combined_waypoints[:, 1],
    }))


combined_generated_waypoints = pd.concat(all_generated_waypoints)
num_train_waypoints = (
  combined_generated_waypoints['type'].values == 'train').sum()
num_generated_waypoints = (
  combined_generated_waypoints['type'].values == 'generated').sum()
frac_train_waypoints = (
  combined_generated_waypoints['type'].values == 'train').mean()
print(f'\nNum train waypoints: {num_train_waypoints}')
print(f'Num generated waypoints: {num_generated_waypoints}')
print(f'Fraction train waypoints: {frac_train_waypoints:.3f}')

if store_generated_waypoints:
  save_path = waypoints_folder / (
    mode + '_' + str(grid_version) + '_' + str(
      float(config['min_distance_to_known'])) + '_' + str(
        float(config['max_distance_to_known'])) + '_' + str(
          config['generate_inner_waypoints']) + '_' + str(
            config['generate_edge_waypoints']) + '_' + str(
              float(config['wall_point_distance_multiplier'])) + '_' + str(
                float(config['inner_point_distance_multiplier'])) + '.csv')
  combined_generated_waypoints.to_csv(save_path, index=False)