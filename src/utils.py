import pickle
import re
from dataclasses import dataclass
import pathlib
from pathlib import Path
import shutil
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import os

from PIL import Image

REPO_DATA_FOLDER = Path(__file__).parent.absolute().parent / "Data files"

CONSIDERED_DATA_FOLDERS = [
    # ADD YOUR DATA FOLDER HERE
    Path("/media/tom/cbd_drive/Kaggle/ILN public/Data"),
    Path(os.environ.get("DATA_PATH", "/home/areeh/data")),
    Path(r"E:\Code\data-ILN"),
    Path("/home/dmitryg/Projects/ILN/data"),
    Path(os.environ.get("DATA_PATH", "./data")),
]


def get_data_folder():
  for f in CONSIDERED_DATA_FOLDERS:
    if f.exists():
      return f

  raise ValueError(
      "Add your data folder to CONSIDERED_DATA_FOLDERS in utils.py")


WIFI_COL_MAP: Dict[int, str] = {
    0: "sys_ts",  # unix time in milliseconds
    1: "ssid",  # name of network shared between access points
    2: "bssid",  # name of physical access point
    3: "rssi",  # signal strength
    4: "frequency",  # frequency we are sending on
    5: "lastseen_ts",
}

TEST_SITES = [
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
]

TEST_FLOOR_MAPPING = {
    "B3": -3,
    "B2": -2,
    "B1": -1,
    "F1": 0,
    "1F": 0,
    "F2": 1,
    "2F": 1,
    "F3": 2,
    "3F": 2,
    "F4": 3,
    "4F": 3,
    "F5": 4,
    "5F": 4,
    "F6": 5,
    "6F": 5,
    "F7": 6,
    "7F": 6,
    "F8": 7,
    "8F": 7,
    "F9": 8,
    "9F": 8,
    "F10": 9,
    "nan": np.nan,
}
NON_TEST_FLOOR_MAPPING = {
    "B": 0,
    "BF": 1,
    "BM": 2,
    "G": 0,
    "M": 0,
    "P1": 0,
    "P2": 1,
    "LG2": -2,
    "LG1": -1,
    "LG": 0,  # This is an outlier (areeh)
    "LM": 0,
    "L1": 1,
    "L2": 2,
    "L3": 3,
    "L4": 4,
    "L5": 5,
    "L6": 6,
    "L7": 7,
    "L8": 8,
    "L9": 9,
    "L10": 10,
    "L11": 11,
}


@dataclass
# Source: https://www.kaggle.com/tvdwiele/indoor-location-exploratory-data-analysis/edit
class ReadData:
  acce: np.ndarray
  acce_uncali: np.ndarray
  gyro: np.ndarray
  gyro_uncali: np.ndarray
  magn: np.ndarray
  magn_uncali: np.ndarray
  ahrs: np.ndarray
  wifi: np.ndarray
  ibeacon: np.ndarray
  waypoint: np.ndarray


def read_data_file(data_filename):
  acce = []
  acce_uncali = []
  gyro = []
  gyro_uncali = []
  magn = []
  magn_uncali = []
  ahrs = []
  wifi = []
  ibeacon = []
  waypoint = []

  with open(data_filename, "r", encoding="utf-8") as file:
    lines = file.readlines()

  added_break_lines = []
  for line_txt in lines:
    line_data = line_txt.strip()

    if not line_data or line_data[0] == "#":
      continue

    type_count = line_data.count("TYPE_")
    if type_count == 1:
      added_break_lines.append(line_txt)
    elif type_count == 0:
      raise ValueError("This should not happen")
    else:
      type_positions = [m.start() for m in re.finditer("TYPE_", line_data)]
      start_pos = 0
      for p_id, p in enumerate(type_positions[1:]):
        end_pos = p - 14
        added_break_lines.append(line_data[start_pos:end_pos])
        start_pos = p - 14

      added_break_lines.append(line_data[start_pos:])

  lines = added_break_lines

  unique_times = []
  for line_id, line_data in enumerate(lines):
    line_data = line_data.strip()

    if not line_data or line_data[0] == "#":
      continue

    line_data = line_data.split("\t")

    new_time = not unique_times or unique_times[-1] != line_data[0]
    if new_time:
      unique_times.append(line_data[0])
    if new_time and len(acce) < len(acce_uncali):
      raise ValueError("This should not happen")

    if line_data[1] == "TYPE_ACCELEROMETER":
      acce.append([
          int(line_data[0]),
          float(line_data[2]),
          float(line_data[3]),
          float(line_data[4]),
      ])
      continue

    if line_data[1] == "TYPE_ACCELEROMETER_UNCALIBRATED":
      acce_uncali.append([
          int(line_data[0]),
          float(line_data[2]),
          float(line_data[3]),
          float(line_data[4]),
      ])
      continue

    if line_data[1] == "TYPE_GYROSCOPE":
      gyro.append([
          int(line_data[0]),
          float(line_data[2]),
          float(line_data[3]),
          float(line_data[4]),
      ])
      continue

    if line_data[1] == "TYPE_GYROSCOPE_UNCALIBRATED":
      gyro_uncali.append([
          int(line_data[0]),
          float(line_data[2]),
          float(line_data[3]),
          float(line_data[4]),
      ])
      continue

    if line_data[1] == "TYPE_MAGNETIC_FIELD":
      magn.append([
          int(line_data[0]),
          float(line_data[2]),
          float(line_data[3]),
          float(line_data[4]),
      ])
      continue

    if line_data[1] == "TYPE_MAGNETIC_FIELD_UNCALIBRATED":
      magn_uncali.append([
          int(line_data[0]),
          float(line_data[2]),
          float(line_data[3]),
          float(line_data[4]),
      ])
      continue

    if line_data[1] == "TYPE_ROTATION_VECTOR":
      ahrs.append([
          int(line_data[0]),
          float(line_data[2]),
          float(line_data[3]),
          float(line_data[4]),
      ])
      continue

    if line_data[1] == "TYPE_WIFI":
      sys_ts = line_data[0]
      ssid = line_data[2]
      bssid = line_data[3]
      rssi = line_data[4]
      frequency = line_data[5]
      lastseen_ts = line_data[6]
      wifi_data = [sys_ts, ssid, bssid, rssi, frequency, lastseen_ts]
      wifi.append(wifi_data)
      continue

    if line_data[1] == "TYPE_BEACON":
      ts = line_data[0]
      uuid = line_data[2]
      major = line_data[3]
      minor = line_data[4]
      rssi = line_data[6]
      ibeacon_data = [ts, "_".join([uuid, major, minor]), rssi]
      ibeacon.append(ibeacon_data)
      continue

    if line_data[1] == "TYPE_WAYPOINT":
      waypoint.append(
          [int(line_data[0]),
           float(line_data[2]),
           float(line_data[3])])

  acce = np.array(acce)
  acce_uncali = np.array(acce_uncali)
  gyro = np.array(gyro)
  gyro_uncali = np.array(gyro_uncali)
  magn = np.array(magn)
  magn_uncali = np.array(magn_uncali)
  ahrs = np.array(ahrs)
  wifi = np.array(wifi)
  ibeacon = np.array(ibeacon)
  waypoint = np.array(waypoint)

  return ReadData(
      acce,
      acce_uncali,
      gyro,
      gyro_uncali,
      magn,
      magn_uncali,
      ahrs,
      wifi,
      ibeacon,
      waypoint,
  )


# Adapted from https://www.kaggle.com/tvdwiele/indoor-location-exploratory-data-analysis/edit
# areeh: that link 404s. Will it work if we're on the same team?
def visualize_trajectory(
    trajectory,
    floor_plan_filename,
    width_meter,
    height_meter,
    title=None,
    mode="lines + markers + text",
    show=False,
):
  fig = go.Figure()

  # add trajectory
  size_list = [6] * trajectory.shape[0]
  size_list[0] = 10
  size_list[-1] = 10

  color_list = ["rgba(4, 174, 4, 0.5)"] * trajectory.shape[0]
  color_list[0] = "rgba(12, 5, 235, 1)"
  color_list[-1] = "rgba(235, 5, 5, 1)"

  position_count = {}
  text_list = []
  for i in range(trajectory.shape[0]):
    if str(trajectory[i]) in position_count:
      position_count[str(trajectory[i])] += 1
    else:
      position_count[str(trajectory[i])] = 0
    text_list.append("        " * position_count[str(trajectory[i])] + f"{i}")
  text_list[0] = "Start Point: 0"
  text_list[-1] = f"End Point: {trajectory.shape[0] - 1}"

  fig.add_trace(
      go.Scattergl(
          x=trajectory[:, 0],
          y=trajectory[:, 1],
          mode=mode,
          marker=dict(size=size_list, color=color_list),
          line=dict(
              shape="linear", color="rgb(100, 10, 100)", width=2, dash="dot"),
          text=text_list,
          textposition="top center",
          name="trajectory",
      ))

  # add floor plan
  floor_plan = Image.open(floor_plan_filename)
  fig.update_layout(images=[
      go.layout.Image(
          source=floor_plan,
          xref="x",
          yref="y",
          x=0,
          y=height_meter,
          sizex=width_meter,
          sizey=height_meter,
          sizing="contain",
          opacity=1,
          layer="below",
      )
  ])

  # configure
  fig.update_xaxes(autorange=False, range=[0, width_meter])
  fig.update_yaxes(
      autorange=False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
  fig.update_layout(
      title=go.layout.Title(
          text=title or "No title.",
          xref="paper",
          x=0,
      ),
      autosize=True,
      width=900,
      height=200 + 900 * height_meter / width_meter,
      template="plotly_white",
  )

  if show:
    fig.show()

  return fig


def load_site_floor(floor_df, recompute_grouped_data=False, test_floor=None):
  data_folder = get_data_folder()
  unique_sites = np.unique(floor_df['site_id'].values)
  unique_floors = np.unique(floor_df['text_level'].values)
  unique_modes = np.unique(floor_df['mode'].values)
  test_data = np.all(pd.isnull(unique_floors))
  assert unique_sites.size == 1
  assert test_data or (unique_floors.size == 1)
  assert unique_modes.size == 1

  site = unique_sites[0]
  floor = unique_floors[0]
  mode = unique_modes[0]
  assert test_data or mode in ['train', 'valid', 'all_train']

  if test_data or mode == 'all_train':
    combined_path = data_folder / 'train' / site / test_floor / (
        mode + '.pickle')
    # combined_path = data_folder / 'train' / site / (
    #   floor + '_' + mode + '.pickle')
  else:
    combined_path = data_folder / 'train' / site / floor / (mode + '.pickle')

  # Recompute if the file names have changed or if the combined does not exist
  # or if recompute_grouped_data is True
  recompute = True
  if not recompute_grouped_data:
    if combined_path.exists():
      with open(combined_path, 'rb') as f:
        combined = pickle.load(f)

      combined_ext = set([c['file_meta'].ext_path for c in combined])
      target_ext = set(floor_df.ext_path)
      recompute = combined_ext != target_ext

  if recompute:
    combined = []
    for ext_id, e in enumerate(floor_df.ext_path):
      data_path = data_folder / (
          str(Path(e).with_suffix('')) + '_reshaped.pickle')
      with open(data_path, 'rb') as f:
        file_data = pickle.load(f)
        file_data['file_meta'] = floor_df.iloc[ext_id]
        combined.append(file_data)

    with open(combined_path, 'wb') as handle:
      pickle.dump(combined, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return combined


def get_loss(preds, actual):
  mpe = np.sqrt(((preds - actual)**2).sum(1)).mean()
  return mpe


def add_near_waypoint_t1(wifi, waypoint_times):
  wifi_last_times = wifi.groupby(
    't1_wifi')['t2_wifi'].transform("max").values
  wifi['most_recent_t2_wifi'] = wifi_last_times

  # Flag the t1 ids that are right before or right after the waypoints
  wifi['near_waypoint_t1'] = False

  unique_t1 = np.unique(wifi_last_times)
  near_waypoint_ids = np.zeros_like(unique_t1, dtype=np.bool)
  for w in waypoint_times:
    t1_rank = (unique_t1 < w).sum()
    for r in [t1_rank, t1_rank-1]:
      if r >= 0 and r < near_waypoint_ids.size:
        near_waypoint_ids[r] = True

  for i in range(near_waypoint_ids.size):
    wifi.loc[wifi_last_times == unique_t1[i], 'near_waypoint_t1'] = (
      near_waypoint_ids[i])
    
  return wifi


def interpolate_wifi_trajectory(
    trajectory, ignore_offset_s, batch_interpolated, file_id=-999):
  wifi = trajectory['wifi']
  waypoint = trajectory['waypoint']
  wifi_t1_times = wifi.t1_wifi.values
  wifi_last_times = wifi.most_recent_t2_wifi.values
  waypoint_times = waypoint.time.values
  assert np.all(np.diff(waypoint_times) > 0)

  if batch_interpolated:
    wifi = add_near_waypoint_t1(wifi, waypoint_times)

  waypoint_interp_x = np.ones(wifi_last_times.size) * waypoint.x_waypoint[0]
  waypoint_interp_y = np.ones(wifi_last_times.size) * waypoint.y_waypoint[0]
  w1 = waypoint_times[0]
  for j, w2 in enumerate(waypoint_times[1:]):
    interp_ids = (wifi_t1_times > w1) & (wifi_t1_times <= w2)
    frac = (wifi_t1_times[interp_ids] - w1) / (w2 - w1)
    waypoint_interp_x[interp_ids] = waypoint.x_waypoint[j] + frac * (
        waypoint.x_waypoint[j + 1] - waypoint.x_waypoint[j])
    waypoint_interp_y[interp_ids] = waypoint.y_waypoint[j] + frac * (
        waypoint.y_waypoint[j + 1] - waypoint.y_waypoint[j])
    w1 = w2
  waypoint_interp_x[wifi_t1_times > waypoint_times[-1]] = (
      waypoint.x_waypoint.values[-1])
  waypoint_interp_y[wifi_t1_times > waypoint_times[-1]] = (
      waypoint.y_waypoint.values[-1])

  # Don't consider interpolated waypoints that are a lot before or a lot
  # after the last waypoint location
  keep_ids = np.ones_like(wifi_last_times, dtype=bool)
  keep_ids[wifi_last_times < (waypoint_times[0] - ignore_offset_s * 1000)] = (
      False)
  keep_ids[wifi_last_times > (waypoint_times[-1] + ignore_offset_s * 1000)] = (
      False)

  wifi_waypoints = wifi.copy()
  wifi_waypoints['waypoint_interp_x'] = waypoint_interp_x
  wifi_waypoints['waypoint_interp_y'] = waypoint_interp_y
  wifi_waypoints = wifi_waypoints[keep_ids]
  # wifi_waypoints = wifi_waypoints.groupby(['bssid_wifi',
  #                                          't2_wifi']).first().reset_index()
  wifi_waypoints['file_id'] = file_id

  return wifi_waypoints


def get_test_waypoint_times(data_folder):
  test_waypoint_times_path = data_folder / "test_waypoint_times.pickle"
  if not test_waypoint_times_path.exists():
    sample_submission = pd.read_csv(data_folder / "sample_submission.csv")

    test_waypoint_times = {}
    for spt in sample_submission.site_path_timestamp.values:
      _, path, ts = spt.split("_")
      if path in test_waypoint_times:
        test_waypoint_times[path].append(int(ts))
      else:
        test_waypoint_times[path] = [int(ts)]

    for k in test_waypoint_times:
      test_waypoint_times[k] = np.array(test_waypoint_times[k])

    with open(test_waypoint_times_path, "wb") as handle:
      pickle.dump(test_waypoint_times, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(test_waypoint_times_path, "rb") as f:
    test_waypoint_times = pickle.load(f)

  return test_waypoint_times


def get_test_fns(data_folder):
  test_fns_ordered_path = data_folder / "test_fns_ordered.pickle"
  if not test_fns_ordered_path.exists():
    sample_submission = pd.read_csv(data_folder / "sample_submission.csv")

    fn_counts = {}
    sites = []
    for spt in sample_submission.site_path_timestamp.values:
      site, fn, _ = spt.split("_")
      if fn in fn_counts:
        fn_counts[fn] += 1
      else:
        sites.append(site)
        fn_counts[fn] = 1

    test_fns_ordered = pd.DataFrame({
        "fn": fn_counts.keys(),
        "site": sites,
        "count": fn_counts.values()
    })
    test_fns_ordered.sort_values(["count", "fn"],
                                 ascending=[False, True],
                                 inplace=True)
    test_fns_ordered.index = np.arange(test_fns_ordered.shape[0])

    with open(test_fns_ordered_path, "wb") as handle:
      pickle.dump(test_fns_ordered, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(test_fns_ordered_path, "rb") as f:
    test_fns_ordered = pickle.load(f)

  return test_fns_ordered


def override_test_floor_errors(submission, debug_test_floor_override=False):
  overrides = {
      '5853ed01a28b1d938e25b2d7': -1,  # 24 obs
      '2379a535c2221d54e8caf6ff': 2,  # 35 obs
      'ac93c263f13fb5eda552ca97': 1,  # 15 obs
      'b37fa5dff7ba5b417031990d': 3,  # 11 obs
      '74e1f3f41374ba181468248d': 1,  # 12 obs
      'e8bf0626e27589d807a9751d': 2,  # 8 obs
      '049bb468e7e166e9d6370002': 2,  # 6 obs
      'db7b0850aed5577702f151c4': 2,  # 25 obs
      '2d17cc2d5c1660c51e00d505': 2,  # 16 obs
      '507749a1187be5b582671c62': 2,  # 10 obs
  }
  
  if debug_test_floor_override:
    print("WARNING: overriding test floors with uncertain floor values")
    print("Are you sure you want to do this???")
    
    additional_overrides = {
      'dd4cbd69218f610f27cf33c8': 3,
      '310990c10f799a9b21cd4ff3': 3,
      'ac93c263f13fb5eda552ca97': 2,
      'bd6637f1fa9e0074495caaca': 6,
      'ac83e97550ad3e7396fb618b': 6,
      '4aaa69d3ff7e13ed5e824aa2': 0,
      'dbc3537c9913889864ebe0c1': 2,
      '346bf8f7f1c62bcb4cd6e307': 3,
      'ed77f28aeaf89b317bc380fa': 3,
      'f7ac4e6a8689647696b64804': 0,
      'bd27a637d58beacf9a51c292': 5,
      '862a4ac32755d252c6948424': 4,
      'e9816721ed502a5414ee6aa4': 1,
      '219d400e61f8e06a066572f8': 0,
      '54b4ec188d91140a6a2ee030': 0,
      '14669681607aa88e65b0d927': 1,
      }
    for k in additional_overrides:
      overrides[k] = additional_overrides[k]

  fns = np.array([sps.split('_')[1] for sps in submission.site_path_timestamp])
  for fn in overrides:
    match_rows = np.where(fns == fn)[0]
    submission.loc[match_rows, 'floor'] = overrides[fn]

  return submission


def get_test_floors(data_folder, debug_test_floor_override=False):
  approx_submission = pd.read_csv(data_folder / "submissions" /
                                  ("submission_cost_minimization.csv"))
  approx_submission = override_test_floor_errors(
    approx_submission, debug_test_floor_override)

  test_floors = {}
  sites = []
  for i, spt in enumerate(approx_submission.site_path_timestamp.values):
    site, fn, _ = spt.split("_")
    if not fn in test_floors:
      sites.append(site)
      test_floors[fn] = approx_submission.floor[i]

  test_floors_path = data_folder / "test_floors.csv"
  if not test_floors_path.exists():
    test_floors_df = pd.DataFrame({
        "site": sites,
        "fn": test_floors.keys(),
        "level": test_floors.values()
    })
    test_floors_df.to_csv(test_floors_path, index=False)

  return test_floors


def convert_to_submission(data_folder, test_preds):
  submission = pd.read_csv(data_folder / "submissions" /
                           ("submission_snap_to_grid.csv"))

  for i, spt in enumerate(submission.site_path_timestamp.values):
    site, fn, ts = spt.split("_")
    floor_pred, x_pred, y_pred = test_preds[site, fn, int(ts)]
    submission.at[i, "floor"] = floor_pred
    submission.at[i, "x"] = x_pred
    submission.at[i, "y"] = y_pred

  return submission


def get_train_waypoints(data_folder, df):
  train_waypoints_path = data_folder / "train_waypoints.pickle"
  if not train_waypoints_path.exists():
    all_waypoints = []
    waypoint_counts = {}
    for i in range([100, df.shape[0]][1]):
      print(i)
      if (df["mode"][i] == "train") and df.test_site[i]:
        pickle_path = data_folder / (
            str(Path(df.ext_path[i]).with_suffix("")) + "_reshaped.pickle")
        with open(pickle_path, "rb") as f:
          trajectory = pickle.load(f)

        site = df.site_id[i]
        floor = df.level[i]
        for w_id in range(trajectory["waypoint"].shape[0]):
          x = trajectory["waypoint"]["x_waypoint"][w_id]
          y = trajectory["waypoint"]["y_waypoint"][w_id]
          k = (site, floor, x, y)
          if k in waypoint_counts:
            waypoint_counts[k] += 1
          else:
            waypoint_counts[k] = 1

          fn = Path(df["ext_path"][i]).stem
          all_waypoints.append(tuple([fn] + list(k)))

    train_waypoints = pd.DataFrame(np.array(all_waypoints))
    train_waypoints.columns = ["fn", "site_id", "level", "x", "y"]
    train_waypoints = train_waypoints.astype({
        "level": float,
        "x": float,
        "y": float,
    })
    data = {
        "train_waypoints": train_waypoints,
        "waypoint_counts": waypoint_counts,
    }

    with open(train_waypoints_path, "wb") as handle:
      pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(train_waypoints_path, "rb") as f:
    data = pickle.load(f)

  train_waypoints = data["train_waypoints"]
  waypoint_counts = data["waypoint_counts"]

  return train_waypoints, waypoint_counts


def interpolate_wifi_waypoints(trajectories,
                               ignore_offset_s=2,
                               recompute=False,
                               batch_interpolated=False):
  for i in range(len(trajectories)):
    if not 'wifi_waypoints' in trajectories[i] or recompute:
      trajectories[i]['wifi_waypoints'] = interpolate_wifi_trajectory(
          trajectories[i], ignore_offset_s, batch_interpolated, i)
    else:
      trajectories[i]['wifi_waypoints']['file_id'] = i


def aggregate_wifi_near_time(trajectories, time_window):
  for i, t in enumerate(trajectories):
    # print(i)
    wifi = t['wifi']
    
    if not 'near_waypoint_t1' in wifi.columns:
      wifi = add_near_waypoint_t1(wifi, t['waypoint_times'])
    
    bssid = {}
    wifi_bssid = wifi.bssid_wifi.values
    wifi_t1s = wifi.t1_wifi.values
    unique_t1 = np.sort(np.unique(wifi_t1s))
    wifi_rssid = wifi.rssid_wifi.values
    wifi_delays = wifi.groupby('t1_wifi')['t2_wifi'].transform(
        "max").values - wifi.t2_wifi.values
    t1_last_t2s = wifi.groupby('t1_wifi')['t2_wifi'].aggregate("max").values
    t1_last_t2s_map = {k: v for (k, v) in zip(unique_t1, t1_last_t2s)}
    t1_near_waypoint = wifi.groupby(
        ['t1_wifi']).first().reset_index().near_waypoint_t1.values
    near_waypoint_t1_map = {k: v for (k, v) in zip(unique_t1, t1_near_waypoint)}
    for j, k in enumerate(wifi_bssid):
      v = (wifi_t1s[j], wifi_delays[j], wifi_rssid[j], j)
      if k in bssid:
        bssid[k].append(v)
      else:
        bssid[k] = [v]

    bssid = {k: np.array(v) for k, v in bssid.items()}

    agg_wifi = []
    for b in bssid:
      b_vals = bssid[b]
      for t in unique_t1:
        time_diffs = np.abs(b_vals[:, 0] - t) / 1000
        valid_time_rows = np.where(time_diffs <= time_window)[0]
        if valid_time_rows.size:
          target_row = valid_time_rows[np.argmax(b_vals[valid_time_rows, 2] -
                                                 1e-9 *
                                                 b_vals[valid_time_rows, 1])]
          agg_wifi.append((b_vals[target_row,
                                  3], t, b_vals[target_row,
                                                1], b_vals[target_row, 2],
                           t1_last_t2s_map[t], near_waypoint_t1_map[t]))

    agg_wifi = np.array(agg_wifi)
    with pd.option_context('mode.chained_assignment', None):
      new_wifi = wifi.iloc[agg_wifi[:, 0]]
      new_wifi.t1_wifi = agg_wifi[:, 1]
      new_wifi.most_recent_t2_wifi = agg_wifi[:, 4]
      new_wifi.t2_wifi = new_wifi.most_recent_t2_wifi - agg_wifi[:, 2]
      new_wifi.near_waypoint_t1 = agg_wifi[:, 5].astype(np.bool)
      new_wifi.rssid_wifi = agg_wifi[:, 3]
      new_wifi.sort_values(["t1_wifi", "rssid_wifi", "bssid_wifi"],
                           ascending=[True, False, True],
                           inplace=True)
      new_wifi.index = np.arange(new_wifi.shape[0])

    trajectories[i]['wifi'] = new_wifi


def group_waypoints_bssid(trajectories):
  combined = pd.concat([t["wifi_waypoints"] for t in trajectories])
  grouped = dict(tuple(combined.groupby("bssid_wifi")))
  for k in grouped.keys():
    grouped[k].index = np.arange(grouped[k].shape[0])

  return grouped


def interpolate_predictions(pos_preds, timestamps):
  pred_times = np.array(list(pos_preds.keys()))
  pred_vals = np.array(list(pos_preds.values()))
  assert np.all(np.diff(pred_times) > 0)

  num_timestamps = timestamps.size
  interpolations = np.zeros((num_timestamps, 2))
  for i in range(num_timestamps):
    ts = timestamps[i]
    if ts <= pred_times[0]:
      interpolations[i] = pred_vals[0]
    elif ts >= pred_times[-1]:
      interpolations[i] = pred_vals[-1]
    else:
      start_id = (ts > pred_times).sum() - 1
      interp_sec = (ts - pred_times[start_id]) / (
          pred_times[start_id + 1] - pred_times[start_id])
      interpolations[i] = pred_vals[start_id] + interp_sec * (
          pred_vals[start_id + 1] - pred_vals[start_id])

  return interpolations


def get_most_freq_np_str(a):
  unique, pos = np.unique(a, return_inverse=True)

  return unique[np.bincount(pos).argmax()]


def interpolate_preds(times, preds, waypoint_times):
  num_preds, pred_dim = preds.shape

  num_waypoints = waypoint_times.size
  interpolated = np.tile(preds[0:1], [num_waypoints, 1])
  for i in range(num_preds - 1):
    waypoint_ids = (waypoint_times > times[i]) & (
        waypoint_times <= times[i + 1])
    if np.any(waypoint_ids):
      fractions = np.expand_dims(
          (waypoint_times[waypoint_ids] - times[i]) / (times[i + 1] - times[i]),
          1)
      interpolated[waypoint_ids] = preds[i] + fractions * np.expand_dims(
          preds[i + 1] - preds[i], 0)

  last_pred_indexes = waypoint_times > times[-1]
  interpolated[last_pred_indexes] = preds[-1:]

  return interpolated

def get_best_opt_error(preds):
  fns = np.unique(preds.fn.values)
  best_opt_error = np.copy(preds.after_optim_error.values)
  for fn in fns:
    fn_rows = np.where(preds.fn.values == fn)[0]
    if preds.all_targets_on_waypoints.values[fn_rows[0]]:
      selected_penalty = preds.selected_total_penalty.values[fn_rows].sum()
      nearest_penalty = preds.nearest_total_penalty.values[fn_rows].sum()
      if nearest_penalty < selected_penalty:
        best_opt_error[fn_rows] = 0
    
  return best_opt_error


def load(path):
  with open(path, 'rb') as f:
    return pickle.load(f)
  
def copy_data_files(data_copy_files):
  data_folder = get_data_folder()
  repo_data_folder = pathlib.Path(__file__).parent.absolute().parent / 'data'
  for f, e in data_copy_files:
    if e:
      target_folder = data_folder / e
    else:
      target_folder = data_folder
    pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)
    target_path = target_folder / f
    if not target_path.is_file():
      source_path = repo_data_folder / f
      shutil.copy(source_path, target_path)
