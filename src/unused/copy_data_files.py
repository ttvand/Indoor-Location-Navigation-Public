import numpy as np

import pandas as pd
import pathlib
from pathlib import Path
import shutil

import utils


data_folder = utils.get_data_folder()
target_data_folder = data_folder.parent.parent / (
  'ILN public') / 'Data'
df = pd.read_csv(data_folder / 'file_summary.csv')

def copy_all_text(source_folder, target_folder, pattern):
  p = Path(source_folder).glob(pattern)
  files = [x for x in p if x.is_file()]
  pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)
  for f in files:
    ext = str(f).split("/")[-1]
    target_path = target_folder / ext
    
    if not target_path.is_file():
      shutil.copy(f, target_path)

# Copy all raw test text files
source_copy_folder = data_folder / 'test'
target_copy_folder = target_data_folder / 'test'
copy_all_text(source_copy_folder, target_copy_folder, '*.txt')

# Copy all test parquet files
source_copy_folder = data_folder / 'reference_preprocessed' / 'test'
target_copy_folder = target_data_folder / 'reference_preprocessed' / 'test' / (
  'test')
copy_all_text(source_copy_folder, target_copy_folder, '*.parquet')

for site in np.unique(df.site_id.values):
  for floor in np.unique(df.text_level.values[
      (df.site_id == site) & (df['mode'] == 'train')]):
    # Copy all raw train text files for the considered floor
    source_copy_folder = data_folder / 'train' / site / floor
    target_copy_folder = target_data_folder / 'train' / site / floor
    copy_all_text(source_copy_folder, target_copy_folder, '*.txt')
    
    # Copy all train parquet files for the considered floor
    source_copy_folder = data_folder / 'reference_preprocessed' / 'train' / (
      site) / floor
    target_copy_folder = target_data_folder / 'reference_preprocessed' / (
      'train') / 'train' / site / floor
    copy_all_text(source_copy_folder, target_copy_folder, '*.parquet')
