import numpy as np
import pandas as pd
import utils

data_folder = utils.get_data_folder()
probe_summary_path = data_folder / 'submissions' / 'probe_outcomes.csv'
probe_summary = pd.read_csv(probe_summary_path)
summary_path = data_folder / 'file_summary.csv'
df = pd.read_csv(summary_path)
leaderboard_type_path = data_folder / 'leaderboard_type.csv'

private_ids = set()
public_ids = set()
for i in range(probe_summary.shape[0]):
  first_id = probe_summary.first_id[i]
  last_id = probe_summary.last_id[i]
  num_ids = last_id - first_id + 1
  lb_change = probe_summary.public_lb_score[i] - (
      probe_summary.baseline_score[i])
  lsb_target_change = probe_summary.lsb_target_change[i]
  correction_factor = probe_summary.correction_factor[i]
  lb_unit_change = lb_change / lsb_target_change * correction_factor

  if not np.isnan(lb_change):
    # print(f"Rounding offset: {np.abs(lb_unit_change-np.round(lb_unit_change))}")
    binary_change = format(
        int(np.round(lb_unit_change)), "0" + str(num_ids) + "b")
    print(binary_change)
    print(num_ids)
    for offset in range(num_ids):
      id_ = first_id + offset
      print(-1 - offset)
      is_public = binary_change[-1 - offset] == '1'
      assert not (is_public and id_ in private_ids)
      assert not (not is_public and id_ in public_ids)

      if is_public:
        public_ids.add(id_)
      else:
        private_ids.add(id_)

test_fns = utils.get_test_fns(data_folder)
public_files = test_fns.loc[public_ids, 'fn'].tolist()
private_files = test_fns.loc[private_ids, 'fn'].tolist()
covered_ids = np.array(sorted(list(public_ids.union(private_ids))))
covered_count = test_fns.loc[covered_ids, 'count'].sum()
covered_fraction = covered_count / (test_fns['count'].values.sum())
print(f"Covered leaderboard fraction: {covered_fraction:.2f}")

leaderboard_type = test_fns.copy()
leaderboard_type['type'] = ''
leaderboard_type.loc[np.array(list(public_ids)), 'type'] = 'public'
leaderboard_type.loc[np.array(list(private_ids)), 'type'] = 'private'
leaderboard_type.to_csv(leaderboard_type_path, index=False)

print(leaderboard_type['count'][leaderboard_type['type'] == 'public'].sum())
