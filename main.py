#!/usr/bin/env python3
import argparse
import sys
sys.path.insert(0, './src')

import apply_correct_sensor_preds
import combine_sensor_data
import create_stratified_holdout_set
import infer_device_type
import infer_start_end_time_leak
import meta_file_preprocessing
import meta_stats_sensor_subtrajectories
import model_correct_sensor_preds
import non_parametric_wifi_model
import prepare_features_correct_sensor_preds
import reshape_reference_preprocessed
import sensor_errors_over_time_device
import utils
import agg_stats
import model_gbm_wifi
import model_sensor_dist
import model_sensor_movement
import model_sensor_movement2

def main(mode, consider_multiprocessing):
  # Preparation of base model dependencies
  utils.copy_data_files()
  meta_file_preprocessing.run()
  reshape_reference_preprocessed.run()
  create_stratified_holdout_set.run()
  combine_sensor_data.run()
  if meta_stats_sensor_subtrajectories.run():
    infer_device_type.run()
    meta_stats_sensor_subtrajectories.run() # Rerun with infered device ids
  
  # Base models
  agg_stats.run()
  model_gbm_wifi.prepare_data()
  model_gbm_wifi.fit('valid') # -> data/gbm_wifi_{mode}_predictions.csv
  model_sensor_dist.run('cv', fast=True) # -> data/sensor_dist_cv
  model_sensor_dist.run(mode, fast=True) # -> data/sensor_dist_{mode}
  model_sensor_movement.run('cv', fast=True) # -> data/sensor_mov_cv
  model_sensor_movement.run(mode, fast=True) # -> data/sensor_mov_{mode}
  model_sensor_movement2.run(mode, fast=True) # -> data/sensor_mov2_{mode}
  non_parametric_wifi_model.run(
    mode, consider_multiprocessing=consider_multiprocessing)
  
  # Other optimization dependencies
  sensor_errors_over_time_device.run()
  if mode == 'test':
    prepare_features_correct_sensor_preds.run('valid')
    model_correct_sensor_preds.run('valid')
  prepare_features_correct_sensor_preds.run(mode)
  model_correct_sensor_preds.run(mode)
  apply_correct_sensor_preds.run(mode)
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
  