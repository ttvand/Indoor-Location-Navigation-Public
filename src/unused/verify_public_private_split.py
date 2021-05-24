import numpy as np
import pandas as pd
import utils

data_folder = utils.get_data_folder()
leaderboard_type_path = data_folder / 'leaderboard_type.csv'
submission_folder = data_folder / 'submissions'
submission_path = submission_folder / 'verify_public_private.csv'
max_submission_path = submission_folder / 'probe_max_submission.csv'

leaderboard_type = pd.read_csv(leaderboard_type_path)
submission = pd.read_csv(max_submission_path)
private_fns = leaderboard_type.fn[leaderboard_type['type'] ==
                                  'private'].tolist()
public_lb = leaderboard_type.iloc[(leaderboard_type['type'] == 'public').values]
private_rows = []

for i, spt in enumerate(submission.site_path_timestamp):
  _, fn, _ = spt.split('_')
  if fn in private_fns:
    private_rows.append(i)

submission.loc[np.array(private_rows), 'floor'] += 1
submission.to_csv(submission_path, index=False)
