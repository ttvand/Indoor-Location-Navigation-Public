#!/usr/bin/env python3
import argparse
import sys
sys.path.insert(0, './src')

import apply_correct_sensor_preds
import combine_predictions_beamsearch
import combine_predictions_beamsearch_multiple_grids
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
import model_sensor_dist
import model_sensor_movement
import model_sensor_movement2
import write_site_time_ordered_waypoints

def main(
    mode, consider_multiprocessing, fast_sensor_models, copy_sensor_models):
  # Preparation of base model dependencies
  utils.copy_data_files()
  if copy_sensor_models:
    utils.copy_sensor_files()
  meta_file_preprocessing.run()
  reshape_reference_preprocessed.run()
  create_stratified_holdout_set.run()
  combine_sensor_data.run()
  if meta_stats_sensor_subtrajectories.run():
    infer_device_type.run()
    meta_stats_sensor_subtrajectories.run() # Rerun with infered device ids
  write_site_time_ordered_waypoints.run()
  
  # Base models
  non_parametric_wifi_model.run(
    mode, consider_multiprocessing=consider_multiprocessing)
  agg_stats.run()
  for m in ['cv', mode, 'valid']:
    model_sensor_dist.run(
      m, fast=fast_sensor_models,
      consider_multiprocessing=consider_multiprocessing)
  for m in ['cv', mode, 'valid']:
    model_sensor_movement.run(
      m, fast=fast_sensor_models,
      consider_multiprocessing=consider_multiprocessing)
  model_sensor_movement2.run(
      mode, fast=fast_sensor_models,
      consider_multiprocessing=consider_multiprocessing)
  
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
  for grid_type in ["walls_only_old", "sparse_inner", "dense_inner"]:
    combine_predictions_beamsearch.run(
      mode, grid_type, consider_multiprocessing)
  
  # Ensembling
  for ensemble_strategy_id in [0, 1]:
    combine_predictions_beamsearch_multiple_grids.run(
      mode, ensemble_strategy_id)
  
  print(f"\nThe pipeline in mode '{mode}' completed successfully!")
  
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--mode", default='test')
  parser.add_argument("-s", action='store_false') # Suppress multiprocessing
  parser.add_argument("-f", action='store_true') # Fast (and bad) sensor model mode
  parser.add_argument("-c", action='store_true') # Copy sensor model option
  args = parser.parse_args()
  
  # args.mode = 'valid'
  # args.c = True
  
  print(f"Running main script in mode '{args.mode}'\n")
  main(args.mode, args.s, args.f, args.c)
  