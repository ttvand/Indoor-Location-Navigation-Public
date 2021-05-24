import numpy as np
import pandas as pd
import pickle

import utils

plot_cols=["x_acce", "y_acce", "z_acce"]
# plot_cols=["x_gyro", "y_gyro", "z_gyro"]
# plot_cols=["x_magn", "y_magn", "z_magn"]
# plot_cols=["x_ahrs", "y_ahrs", "z_ahrs"]
# plot_cols=["x_acce_uncali", "y_acce_uncali", "z_acce_uncali"]
# plot_cols=["x_gyro_uncali", "y_gyro_uncali", "z_gyro_uncali"]
# plot_cols=["a_acce", "a_acc", "a_gyro", "a_gyro_uncali", "a_magn",
#            "a_magn_uncali", "a_ahrs"]
fn = [
  '00ff0c9a71cc37a2ebdd0f05', # Random
  '5db03bc111adb40006afcc75', # Random
  '5dc7b46f1cda3700060314b8', # Random
  '5dc78ab417ffdd0006f11f2e', # Random
  '5daeccede415cd0006629418', # Random
  
  '5de0cd6291ec850006b96780', # Very long (mostly not moving)
  '5de0bacbbbb32e0006603cb4', # Very long (mostly not moving)
  '5de0cd65bbb32e0006603ce7', # Very long (mostly not moving)
  '5de0cd5191ec850006b96776', # Long and long distance covered
  '5de0bd4dbbb32e0006603cc1', # Long and long distance covered
  
  '5de0cd5391ec850006b96778', # Long and long distance covered
  '5d074cc8b53a8d0008dd493b', # Very long distance covered
  '5dce884b5516ad00065f03e3',
  '5dc8e91a17ffdd0006f12ce0',
  '5da9a375df065a00069be745',
  
  '5dc7b45217ffdd0006f1235f',
  '5dc7b46f1cda3700060314b8',
  '5dcf8361878f3300066c6dd9',
  '5dc7c59517ffdd0006f125ba',
  '5dccce38757dea0006080071',
  ][19]
sub_trajectory_id = 0
num_sub_traj = 2

data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
df = pd.read_csv(summary_path)

fn_row = np.where(df.fn == fn)[0][0]
mode = df['mode'].values[fn_row]
site = df['site_id'].values[fn_row]
floor = df['text_level'].values[fn_row]

path_ext = fn + '_reshaped.pickle'
if mode == 'test':
  data_path = data_folder / mode / path_ext
else:
  data_path = data_folder / 'train' / site / floor / path_ext
  
with open(data_path, 'rb') as f:
  file_data = pickle.load(f)
  
waypoint_times = file_data['waypoint_times']
share_time_vals = file_data['shared_time'].time.values
start_time = waypoint_times[sub_trajectory_id]
end_sub_id = min(waypoint_times.size-1, sub_trajectory_id+num_sub_traj)
end_time = waypoint_times[end_sub_id]
start_row = max(0, (share_time_vals <= start_time).sum() - 1)
end_row = min(share_time_vals.size, (
  share_time_vals < end_time).sum()+1)

plot_data = file_data['shared_time'].iloc[np.arange(start_row, end_row)]

ax = plot_data.plot(x="time", y=plot_cols)
# for i in range(waypoint_times.size-1):
#   ax.axvline(waypoint_times[i+1], color="black", linestyle="--")