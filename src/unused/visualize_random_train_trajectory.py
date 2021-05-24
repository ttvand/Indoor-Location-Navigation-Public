import json
from pathlib import Path

import numpy as np
import pandas as pd
import utils

only_consider_test_sites = True
overwrite_preprocessed_files = False
num_random_plots = 10
site_filter = [
    None,
    "5a0546857ecc773753327266",  # 1 - Well aligned, no waypoints inside buildings
    "5c3c44b80379370013e0fd2b",  # 2 - Well aligned, rare waypoints inside buildings. Relation between num waypoints and floor (?).
    "5d27075f03f801723c2e360f",  # 3 - Building misaligned, needs waypoint vertical upwards shift of 3-5 units
    "5d27096c03f801723c31e5e0",  # 4 - Well aligned, no waypoints inside buildings
    "5d27097f03f801723c320d97",  # 5 - Well aligned, no waypoints inside buildings
    "5d27099f03f801723c32511d",  # 6 - Well aligned, no waypoints inside buildings, simple building layout
    "5d2709a003f801723c3251bf",  # 7 - Well aligned, waypoints often at edges of buildings
    "5d2709b303f801723c327472",  # 8 - Well aligned, no waypoints inside buildings
    "5d2709bb03f801723c32852c",  # 9 - Well aligned, no waypoints inside buildings
    "5d2709c303f801723c3299ee",  # 10 - Well aligned, no waypoints inside buildings - this site consists of 2/3 disconnected areas on the lower floors and the mid high floors
    "5d2709d403f801723c32bd39",  # 11 - Well aligned, rare waypoints inside buildings. Circular layout.
    "5d2709e003f801723c32d896",  # 12 - Well aligned, rare waypoints inside buildings
    "5da138274db8ce0c98bbd3d2",  # 13 - Well aligned, rare waypoints inside buildings - cool shape!
    "5da1382d4db8ce0c98bbe92e",  # 14 - Well aligned, no waypoints inside buildings
    "5da138314db8ce0c98bbf3a0",  # 15 - Well aligned, occasional waypoints at edge of buildings
    "5da138364db8ce0c98bc00f1",  # 16 - Well aligned, occasional waypoints at edge of buildings
    "5da1383b4db8ce0c98bc11ab",  # 17 - Well aligned, no waypoints inside buildings
    "5da138754db8ce0c98bca82f",  # 18 - Well aligned, waypoints often at edges of buildings, sometimes inside buildings
    "5da138764db8ce0c98bcaa46",  # 19 - Well aligned, no waypoints inside buildings
    "5da1389e4db8ce0c98bd0547",  # 20 - Well aligned, no waypoints inside buildings. Some open areas seem unaccessible
    "5da138b74db8ce0c98bd4774",  # 21 - Well aligned, no waypoints inside buildings
    "5da958dd46f8266d0737457b",  # 22 - Well aligned, rare waypoints inside buildings
    "5dbc1d84c1eb61796cf7c010",  # 23 - Well aligned, no waypoints inside buildings
    "5dc8cea7659e181adb076a3f",  # 24 - Well aligned, no waypoints inside buildings
][4]
level_filter = [None, "F1", -1][2]
target_fn = [None, "5dd5290c50e04e0006f5651e", "5dd5216b50e04e0006f56476"][0]

data_folder = utils.get_data_folder()
summary_path = data_folder / "file_summary.csv"
if not "df" in locals():
  df = pd.read_csv(summary_path)

considered_ids = np.where(df["mode"] == "train")[0]
if only_consider_test_sites:
  considered_ids = considered_ids[df.test_site[considered_ids]]
if site_filter is not None:
  considered_ids = considered_ids[df.site_id[considered_ids] == site_filter]
if level_filter is not None:
  if isinstance(level_filter, int):
    considered_ids = considered_ids[df.level[considered_ids] == level_filter]
  else:
    considered_ids = considered_ids[df.text_level[considered_ids] ==
                                    level_filter]
if target_fn is not None:
  considered_ids = considered_ids[df.fn[considered_ids] == target_fn]


def visualize_train_trajectory(meta_info, data_path):
  _id, floor, fn = Path(data_path).parts[-3:]

  train_floor_data = utils.read_data_file(data_path)
  data_folder = Path(data_path).parents[3]
  floor_folder = data_folder / f"metadata/{_id}/{floor}"
  floor_info_path = floor_folder / "floor_info.json"
  floor_image_path = floor_folder / "floor_image.png"
  with open(floor_info_path) as f:
    train_floor_info = json.load(f)

  fig = utils.visualize_trajectory(
      train_floor_data.waypoint[:, 1:3],
      floor_image_path,
      train_floor_info["map_info"]["width"],
      train_floor_info["map_info"]["height"],
  )

  plot_trajectory_folder = data_folder / "train_trajectories"
  Path(plot_trajectory_folder).mkdir(parents=True, exist_ok=True)
  save_path = plot_trajectory_folder / (fn[:-4] + ".html")
  fig.write_html(str(save_path))

# Plot the trajectory for a random training id
for _ in range(num_random_plots):
  target_row = np.random.choice(considered_ids)
  data_path = data_folder / df.ext_path[target_row]
  visualize_train_trajectory(df.iloc[target_row], data_path)
