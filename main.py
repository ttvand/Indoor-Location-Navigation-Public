#!/usr/bin/env python3
import argparse
import sys
sys.path.insert(0, './src')

import combine_sensor_data
import create_stratified_holdout_set
import infer_device_type
import infer_start_end_time_leak
import meta_file_preprocessing
import meta_stats_sensor_subtrajectories
import non_parametric_wifi_model
import reshape_reference_preprocessed
import sensor_errors_over_time_device
import utils

def main(mode, consider_multiprocessing):
  # Preparation of base model dependencies
  utils.copy_data_files()
  meta_file_preprocessing.run()
  reshape_reference_preprocessed.run()
  create_stratified_holdout_set.run()
  combine_sensor_data.run()
  meta_stats_sensor_subtrajectories.run()
  infer_device_type.run()
  meta_stats_sensor_subtrajectories.run() # Rerun with infered device ids
  
  # Base models
  non_parametric_wifi_model.run(
    mode, consider_multiprocessing=consider_multiprocessing)
  import pdb; pdb.set_trace()
  x=1
  
  # Other optimization dependencies
  sensor_errors_over_time_device.run()
  infer_start_end_time_leak.run(mode)
  
  # Optimization
  import pdb; pdb.set_trace()
  x=1
  
  # Ensembling
  import pdb; pdb.set_trace()
  x=1
  
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--mode", default='test')
  parser.add_argument("-s", action='store_false')
  args = parser.parse_args()
  # args.mode = 'valid'
  
  main(args.mode, args.s)
  