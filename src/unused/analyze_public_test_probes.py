import numpy as np
import pandas as pd

import utils

data_folder = utils.get_data_folder()
feedback_path = data_folder / 'leaderboard feedback.txt'
leaderboard_types_path = data_folder / 'leaderboard_type.csv'

if not 'submissions_df' in locals():
  leaderboard_types = pd.read_csv(leaderboard_types_path)
  
  # import os
  # lb_outcomes_raw = os.popen(
  #   "kaggle competitions submissions -c indoor-location-navigation -v").read()
  with open(feedback_path) as f:
    lines = f.readlines()
    
  submission_name = ''
  delay = ''
  comment = ''
  score = None
  
  id_ = 0
  submissions = []
  for l in lines[:-6]:
    if id_ == 0:
      submission_name += l[:-1]
      if submission_name[-3:] == 'csv':
        id_ = 1
      else:
        import pdb; pdb.set_trace()
        x=1
    elif id_ == 1:
      delay = l[:-1]
      id_ = 2
    elif id_ == 2:
      if l != "\n":
        comment = l[:-1]
        id_ = 3
    elif id_ == 3:
      try:
        score = float(l[:-1])
      except:
        assert l[:-1] == "Error"
      id_ = 4
    elif id_ == 4:
      assert l == "\n"
      
      if score is not None:
        submissions.append({
          'name': submission_name,
          'delay': delay,
          'comment': comment,
          'score': score,
          })
      
      submission_name = ''
      delay = ''
      comment = ''
      score = None
      id_ = 0
      
  submissions_df = pd.DataFrame(submissions)
  
public_test_fn_rows = np.where(leaderboard_types['type'].values == 'public')[0]
probe_summary = []
for r in public_test_fn_rows:
  fn = leaderboard_types.fn.values[r]
  count = leaderboard_types['count'].values[r]
  
  snap_grid_id = None
  apr_11_id = None
  apr_29_id = np.nan
  for i, n in enumerate(submissions_df.name.values):
    if n[:10] == "probe_snap" and (
        n[-32:] == "submission_cost_minimization.csv") and (fn in n):
      assert snap_grid_id is None
      snap_grid_id = i
      
    if ('inflated - test - 2021-04-11 123039' in n) and (fn in n) and (
        'submission_cost_minimization.csv' in n):
      assert apr_11_id is None
      apr_11_id = i
      
    if ('inflated - test - 2021-04-29 194551' in n) and (fn in n) and (
        'submission_cost_minimization.csv' in n):
      assert np.isnan(apr_29_id)
      apr_29_id = i

  assert snap_grid_id is not None
  assert apr_11_id is not None
  
  snap_grid_score_change = submissions_df.score.values[snap_grid_id] - 600
  apr_11_score_change = submissions_df.score.values[apr_11_id] - 600
  apr_29_score_change = np.nan if np.isnan(apr_29_id) else (
    submissions_df.score.values[apr_29_id] - 600)
  
  probe_summary.append({
    'fn': fn,
    'count': count,
    'total_snap_grid_score_change': snap_grid_score_change,
    'total_apr_11_score_change': apr_11_score_change,
    'total_apr_29_score_change': apr_29_score_change,
    'snap_grid_score_av_change': snap_grid_score_change*1527/count,
    'apr_11_score_av_change': apr_11_score_change*1527/count,
    'apr_29_score_av_change': apr_29_score_change*1527/count,
    'best_change_ref': min([0, snap_grid_score_change, apr_11_score_change]),
    })
probe_summary_df = pd.DataFrame(probe_summary)
# probe_summary_df.to_csv(data_folder / 'probe_summary.csv', index=False)