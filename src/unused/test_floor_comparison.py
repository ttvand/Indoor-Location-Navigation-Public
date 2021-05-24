import numpy as np
import pandas as pd
import pathlib

import utils

source_public_notebook = 'submission_cost_minimization.csv'
tom_floor_submission = 'test - 2021-05-12 12:34:45.csv'
dmitry_floor_preds = 'lgbm_floor_pred_v1_test.csv'

data_folder = utils.get_data_folder()
submissions_folder = data_folder / 'submissions'
repo_data_folder = pathlib.Path(__file__).parent.absolute().parent / (
  'Data files')

leaderboard_types = pd.read_csv(data_folder / 'leaderboard_type.csv')
leaderboard_types.sort_values(['fn'], inplace=True)

public_pred_path = submissions_folder / source_public_notebook
public_submission = pd.read_csv(public_pred_path)
public_submission['fn'] = [sps.split('_')[1] for sps in (
  public_submission.site_path_timestamp.values)]
public_floors = (public_submission.groupby('fn').first().reset_index())[[
  'fn', 'floor']]
public_floors.sort_values(['fn'], inplace=True)
public_preds = public_floors.floor.values

tom_pred_path = submissions_folder / tom_floor_submission
tom_submission = pd.read_csv(tom_pred_path)
tom_submission['fn'] = [sps.split('_')[1] for sps in (
  tom_submission.site_path_timestamp.values)]
tom_floors = (tom_submission.groupby('fn').first().reset_index())[[
  'fn', 'floor']]
tom_floors.sort_values(['fn'], inplace=True)
tom_preds = tom_floors.floor.values

dmitry_pred_path = repo_data_folder / dmitry_floor_preds
dmitry_floors = pd.read_csv(dmitry_pred_path)
dmitry_floors = dmitry_floors[['fn', 'preds']]
dmitry_floors.sort_values(['fn'], inplace=True)
dmitry_floors.index = np.arange(dmitry_floors.shape[0])
dmitry_preds = dmitry_floors.preds.values

all_preds = pd.DataFrame({
  'fn': leaderboard_types.fn.values,
  'site': leaderboard_types.site.values,
  'count': leaderboard_types['count'].values,
  'type': leaderboard_types['type'].values,
  'public_pred': public_preds,
  'tom_pred': tom_preds,
  'dmitry_pred': dmitry_preds,
  'agree_tom_public': tom_preds == public_preds,
  'agree_dmitry_public': dmitry_preds == public_preds,
  'agree_tom_dmitry': tom_preds == dmitry_preds,
  })
all_preds.sort_values(['agree_tom_dmitry', 'agree_tom_public'], inplace=True)

save_path = data_folder / 'test_floor_comparison.csv'
all_preds.to_csv(save_path, index=False)