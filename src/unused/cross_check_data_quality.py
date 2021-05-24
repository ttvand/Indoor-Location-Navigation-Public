import dataclasses
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import utils

only_check_test_sites = True

FIELD_MAPPING = {
    'TYPE_ACCELEROMETER': 'acce',
    'TYPE_ACCELEROMETER_UNCALIBRATED': 'acce_uncali',
    'TYPE_GYROSCOPE': 'gyro',
    'TYPE_GYROSCOPE_UNCALIBRATED': 'gyro_uncali',
    'TYPE_MAGNETIC_FIELD': 'magn',
    'TYPE_MAGNETIC_FIELD_UNCALIBRATED': 'magn_uncali',
    'TYPE_ROTATION_VECTOR': 'ahrs',
    'TYPE_WIFI': 'wifi',
    'TYPE_BEACON': 'ibeacon',
    'TYPE_WAYPOINT': 'waypoint',
}


def compare_files(pickle_path: Path, parquet_path: Path):
  with open(pickle_path, 'rb') as f:
    pickle_data = dataclasses.asdict(pickle.load(f))
  parquet_data = pd.read_parquet(parquet_path, engine='fastparquet')
  types = parquet_data.column2

  for k, v in FIELD_MAPPING.items():
    parquet_vals = parquet_data.iloc[(types == k).values].values
    pickle_vals = pickle_data[v]

    try:
      assert parquet_vals.shape[0] >= pickle_vals.shape[0]
      if parquet_vals.shape[0] > pickle_vals.shape[0]:
        print("More data parquet {}".format(Path(pickle_path).stem))
        if pickle_path.exists():
          os.remove(pickle_path)
        continue
      if pickle_vals.size:
        assert np.all(
            parquet_vals[:,
                         0].astype(np.int64) == pickle_vals[:,
                                                            0].astype(np.int64))
      if v == 'wifi':
        if pickle_vals.size:
          assert np.all(parquet_vals[:, 2:4] == pickle_vals[:, 1:3])
          assert np.all(parquet_vals[:, 4].astype(np.int64) ==
                        pickle_vals[:, 3].astype(np.int64))
          assert np.all(parquet_vals[:, 6].astype(np.int64) ==
                        pickle_vals[:, 5].astype(np.int64))
      elif v == 'ibeacon':
        if pickle_vals.size:
          assert np.all(
              pd.Series(pickle_vals[:,
                                    1]).str.split('_', expand=True).values == (
                                        parquet_vals[:, 2:5]))
          assert np.all(parquet_vals[:, 6] == pickle_vals[:, 2])
      else:
        num_val_cols = 2 if v == 'waypoint' else 3
        if pickle_vals.size:
          assert np.all(
              np.isclose(pickle_vals[:, 1:],
                         parquet_vals[:,
                                      2:2 + num_val_cols].astype(np.float32)))
    except Exception as exception:
      import pdb
      pdb.set_trace()
      print(type(exception).__name__)


data_folder = utils.get_data_folder()
parquet_folder = data_folder / 'reference_preprocessed'
summary_path = data_folder / 'file_summary.csv'
df = pd.read_csv(summary_path)

# Loop over all file paths and compare the parquet and pickle files one by one
for i in range(df.shape[0]):
  print(i)
  if not only_check_test_sites or df.test_site[i]:
    pickle_path = data_folder / Path(df.ext_path[i]).with_suffix('.pickle')
    parquet_path = parquet_folder / Path(df.ext_path[i]).with_suffix('.parquet')
    if pickle_path.exists():
      compare_files(pickle_path, parquet_path)
