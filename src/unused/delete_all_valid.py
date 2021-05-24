import numpy as np

import pandas as pd
from pathlib import Path

import utils


data_folder = utils.get_data_folder()
df = pd.read_csv(data_folder / 'file_summary.csv')

def delete_if_exists(p):
  file = Path(p)
  if file.exists():
    Path(p).unlink()

# Delete all validation related files in order to make sure that there is no
# leakage when deleting validation data
delete_if_exists(data_folder / 'holdout_ids.csv')
delete_if_exists(data_folder / 'train_waypoints_timed.csv')
delete_if_exists(data_folder / 'valid_edge_positions.csv')
delete_if_exists(data_folder / 'sensor_data' / 'meta.csv')
delete_if_exists(data_folder / 'sensor_data' / 'train.pickle')
delete_if_exists(data_folder / 'sensor_data' / 'valid.pickle')
for site in np.unique(df.site_id.values):
  for floor in np.unique(df.text_level.values[
      (df.site_id == site) & (df['mode'] == 'train')]):
    delete_if_exists(data_folder / 'train' / site / floor / 'train.pickle')
    delete_if_exists(data_folder / 'train' / site / floor / 'valid.pickle')
    
pth = Path(data_folder / 'train_valid_waypoints')
for child in pth.glob('*'):
  delete_if_exists(child)