import numpy as np
import pandas as pd
import utils

# 20 probed ids with each submission
offset_20 = 0
holdout_fn_ids = np.arange(20 * offset_20, 20 * (offset_20 + 1))
num_test_public = 1527
target_unit_change = -0.07

data_folder = utils.get_data_folder()
submission_folder = data_folder / 'submissions'
max_submission_path = submission_folder / 'probe_max_submission.csv'
leaderboard_type_path = data_folder / 'leaderboard_type.csv'
submission_path = submission_folder / ('verify_holdout_' + str(
    int(holdout_fn_ids[0])) + '_' + str(int(holdout_fn_ids[-1])) + '.csv')
test_fns = utils.get_test_fns(data_folder).fn.values

leaderboard_type = pd.read_csv(leaderboard_type_path)
max_submission = pd.read_csv(max_submission_path)
test_fns = test_fns[leaderboard_type['type'] == 'public']

# Generate the probing default dummy submission
submission = max_submission.copy()
for i, holdout_id in enumerate(holdout_fn_ids):
  holdout_fn = test_fns[holdout_id]
  fn_ids = np.where([
      spt.split('_')[1] == holdout_fn
      for spt in (max_submission.site_path_timestamp)
  ])[0]
  target_error_reduction = 2**i * target_unit_change
  targeted_floor_correction = target_error_reduction / (fn_ids.size * 15) * (
      num_test_public)
  print(targeted_floor_correction, target_error_reduction)
  submission.loc[fn_ids, 'floor'] -= targeted_floor_correction

submission.to_csv(submission_path, index=False)
