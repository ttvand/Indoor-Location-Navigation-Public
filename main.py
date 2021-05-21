#!/usr/bin/env python3
import argparse
import sys
sys.path.insert(0, './src')

import meta_file_preprocessing
import utils

data_copy_files = [
    ('sample_submission.csv', ''),
    ('submission_cost_minimization.csv', 'submissions'),
    ]

def main(mode):
  # Copy the required data files to the data folder
  utils.copy_data_files(data_copy_files)
  
  # Preparation of base model dependencies
  import pdb; pdb.set_trace()
  meta_file_preprocessing.run()
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--mode", default='test')
  args = parser.parse_args()
  main(args.mode)
  
  
  
  
  
  
  
  
  
