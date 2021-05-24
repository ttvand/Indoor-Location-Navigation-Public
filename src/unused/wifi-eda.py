import pickle
from pathlib import Path
from random import choice

import numpy as np
import pandas as pd

from Logic.utils import WIFI_COL_MAP, get_data_folder

test_dir = Path(get_data_folder()) / "test"
pths = list(test_dir.glob('*.pickle'))
sample_file = choice(pths)
data = pickle.load(open(sample_file, 'rb'))
df = pd.DataFrame({v: data.wifi[:, k] for k, v in WIFI_COL_MAP.items()})
updates_per_ts = np.unique(df["sys_ts"].astype(int), return_counts=True)
min_upd_ts = updates_per_ts[1].min()
max_upd_ts = updates_per_ts[1].max()
std_upd_ts = updates_per_ts[1].std()
print(f"Minimum wifi updates that occurred in a timestamp: {min_upd_ts}")
print(f"Maximum wifi updates that occurred in a timestamp: {max_upd_ts}")
print(f"std wifi updates that occurred in a timestamp: {std_upd_ts}")

update_time_diff = df["sys_ts"].astype(int).diff()
update_time_diff = update_time_diff[update_time_diff > 0]
mean = update_time_diff.mean()
std = update_time_diff.std()
print(f"mean time between wifi updates: {mean} ms")
print(f"std: {std} ms")
print()

uniques = df.nunique()
uniques["number_of_entries"] = len(df)
n_ssid = uniques["ssid"]
n_bssid = uniques["bssid"]
print(uniques)

print(df)
