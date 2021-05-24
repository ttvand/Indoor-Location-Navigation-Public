import pickle
from pathlib import Path

import pandas as pd
import utils

only_process_test_sites = True
overwrite_preprocessed_files = False


def preprocess_file(save_path: Path):
  data_file = utils.read_data_file(save_path.with_suffix('.txt'))
  with open(save_path, 'wb') as handle:
    pickle.dump(data_file, handle, protocol=pickle.HIGHEST_PROTOCOL)


data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
df = pd.read_csv(summary_path)

# Loop over all files and convert them one by one to a training compatible
# format which is stored to disk
for i in range(df.shape[0]):
  print(i)
  if not only_process_test_sites or df.test_site[i]:
    save_path = data_folder / Path(df.ext_path[i]).with_suffix('.pickle')
    if overwrite_preprocessed_files or not save_path.exists():
      preprocess_file(save_path)
