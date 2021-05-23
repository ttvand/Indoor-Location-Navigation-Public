import json
import os
import re
import pickle
import itertools
from collections import defaultdict
from pprint import pprint
from typing import Optional, Dict, List

import attr
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import datetime

from Logic.utils import get_data_folder

ts_conv = np.vectorize(datetime.datetime.fromtimestamp)  # ut(10 digit) -> date

# pandas settings -----------------------------------------
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:,.5f}".format

# Graph drawing -------------------------------------------
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib_venn import venn2, venn2_circles
from matplotlib import animation as ani
from IPython.display import Image
from pylab import imread

plt.rcParams["patch.force_edgecolor"] = True
import seaborn as sns

sns.set(style="whitegrid", palette="muted", color_codes=True)
sns.set_style("whitegrid", {"grid.linestyle": "--"})
red = sns.xkcd_rgb["light red"]
green = sns.xkcd_rgb["medium green"]
blue = sns.xkcd_rgb["denim blue"]

# ML -------------------------------------------
from sklearn.preprocessing import LabelEncoder


def unpickle(filename):
  with open(filename, "rb") as fo:
    p = pickle.load(fo)
  return p


def to_pickle(filename, obj):
  with open(filename, "wb") as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


@attr.s(auto_attribs=True, slots=True)
class SiteInfo:
  site: str
  floor: str
  input_path: Path

  # data
  floor_info: Optional[Dict] = None
  site_height: Optional[float] = None
  site_width: Optional[float] = None

  geojson_map: Optional[Dict] = None
  map_type: Optional[str] = None
  features: Optional[Dict] = None

  floor_coordinates: Optional[List] = None
  store_coordinates: Optional[List] = None

  # plot
  x_size: Optional[float] = None
  y_size: Optional[float] = None

  def __attrs_post_init__(self):
    assert self.input_path.exists(
    ), f"input_path do not exist: {self.input_path}"

  def get_site_info(self, keep_raw=False):
    floor_info_path = (
        f"{self.input_path}/metadata/{self.site}/{self.floor}/floor_info.json")
    with open(floor_info_path, "r") as f:
      self.floor_info = json.loads(f.read())
      self.site_height = self.floor_info["map_info"]["height"]
      self.site_width = self.floor_info["map_info"]["width"]
      if not keep_raw:
        del self.floor_info

    geojson_map_path = (
        f"{self.input_path}/metadata/{self.site}/{self.floor}/geojson_map.json")
    with open(geojson_map_path, "r") as f:
      self.geojson_map = json.loads(f.read())
      self.map_type = self.geojson_map["type"]
      self.features = self.geojson_map["features"]

      self.floor_coordinates = self.features[0]["geometry"]["coordinates"]
      self.store_coordinates = [
          self.features[i]["geometry"]["coordinates"]
          for i in range(1, len(self.features))
      ]

      if not keep_raw:
        del self.geojson_map

  def show_site_image(self):
    path = f"{self.input_path}/metadata/{self.site}/{self.floor}/floor_image.png"
    plt.imshow(imread(path), extent=[0, self.site_width, 0, self.site_height])

  def draw_polygon(self, size=8, only_floor=False):

    fig = plt.figure()
    ax = plt.subplot(111)

    xmax, xmin, ymax, ymin = self._draw(
        self.floor_coordinates, ax, calc_minmax=True)
    if not only_floor:
      self._draw(self.store_coordinates, ax, fill=True)
    plt.legend([])

    xrange = xmax - xmin
    yrange = ymax - ymin
    ratio = yrange / xrange

    self.x_size = size
    self.y_size = size * ratio

    fig.set_figwidth(size)
    fig.set_figheight(size * ratio)
    # plt.show()
    return ax

  def _draw(self, coordinates, ax, fill=False, calc_minmax=False):
    xmax, ymax = -np.inf, -np.inf
    xmin, ymin = np.inf, np.inf
    for i in range(len(coordinates)):
      ndim = np.ndim(coordinates[i])
      if ndim == 2:
        corrd_df = pd.DataFrame(coordinates[i])
        if fill:
          ax.fill(corrd_df[0], corrd_df[1], alpha=0.7)
        else:
          corrd_df.plot.line(x=0, y=1, style="-", ax=ax)

        if calc_minmax:
          xmax = max(xmax, corrd_df[0].max())
          xmin = min(xmin, corrd_df[0].min())

          ymax = max(ymax, corrd_df[1].max())
          ymin = min(ymin, corrd_df[1].min())
      elif ndim == 3:
        for j in range(len(coordinates[i])):
          corrd_df = pd.DataFrame(coordinates[i][j])
          if fill:
            ax.fill(corrd_df[0], corrd_df[1], alpha=0.6)
          else:
            corrd_df.plot.line(x=0, y=1, style="-", ax=ax)

          if calc_minmax:
            xmax = max(xmax, corrd_df[0].max())
            xmin = min(xmin, corrd_df[0].min())

            ymax = max(ymax, corrd_df[1].max())
            ymin = min(ymin, corrd_df[1].min())
      else:
        assert False, f"ndim of coordinates should be 2 or 3: {ndim}"
    if calc_minmax:
      return xmax, xmin, ymax, ymin
    else:
      return None


@attr.s(auto_attribs=True, slots=True)
class FeatureStore:
  site: str
  floor: str
  path_id: str

  input_path: Path = get_data_folder()
  save_path: Path = get_data_folder() / "eda_data"
  n_floor: int = attr.ib()
  site_info: SiteInfo = attr.ib()
  meta_info_len: Optional[int] = None
  meta_info_df: Optional[pd.DataFrame] = None

  accelerometer: Optional[pd.DataFrame] = None
  accelerometer_uncalibrated: Optional[pd.DataFrame] = None
  beacon: Optional[pd.DataFrame] = None
  gyroscope: Optional[pd.DataFrame] = None
  gyroscope_uncalibrated: Optional[pd.DataFrame] = None
  magnetic_field: Optional[pd.DataFrame] = None
  magnetic_field_uncalibrated: Optional[pd.DataFrame] = None
  rotation_vector: Optional[pd.DataFrame] = None
  waypoint: Optional[pd.DataFrame] = None
  wifi: Optional[pd.DataFrame] = None

  @n_floor.default
  def n_floor_default(self) -> int:
    return self.floor_convert[self.floor]

  @site_info.default
  def site_info_default(self) -> SiteInfo:
    return SiteInfo(
        site=self.site, floor=self.floor, input_path=self.input_path)

  def __attrs_post_init__(self):
    self.site = self.site.strip()
    self.floor = self.floor.strip()
    self.path_id = self.path_id.strip()

    assert self.input_path.exists(
    ), f"input_path do not exist: {self.input_path}"

    Path(self.save_path).mkdir(parents=True, exist_ok=True)

  floor_convert = {
      "1F": 0,
      "2F": 1,
      "3F": 2,
      "4F": 3,
      "5F": 4,
      "6F": 5,
      "7F": 6,
      "8F": 7,
      "9F": 8,
      "B": -1,
      "B1": -1,
      "B2": -2,
      "B3": -3,
      "BF": -1,
      "BM": -1,
      "F1": 0,
      "F2": 1,
      "F3": 2,
      "F4": 3,
      "F5": 4,
      "F6": 5,
      "F7": 6,
      "F8": 7,
      "F9": 8,
      "F10": 9,
      "L1": 0,
      "L2": 1,
      "L3": 2,
      "L4": 3,
      "L5": 4,
      "L6": 5,
      "L7": 6,
      "L8": 7,
      "L9": 8,
      "L10": 9,
      "L11": 10,
      "G": 0,
      "LG1": 0,
      "LG2": 1,
      "LM": 0,
      "M": 0,
      "P1": 0,
      "P2": 1,
  }

  df_types = [
      "accelerometer",
      "accelerometer_uncalibrated",
      "beacon",
      "gyroscope",
      "gyroscope_uncalibrated",
      "magnetic_field",
      "magnetic_field_uncalibrated",
      "rotation_vector",
      "waypoint",
      "wifi",
  ]

  # https://github.com/location-competition/indoor-location-competition-20
  df_type_cols = {
      "accelerometer": ["timestamp", "x", "y", "z", "accuracy"],
      "accelerometer_uncalibrated": [
          "timestamp",
          "x",
          "y",
          "z",
          "x2",
          "y2",
          "z2",
          "accuracy",
      ],
      "beacon": [
          "timestamp",
          "uuid",
          "major_id",
          "minor_id",
          "tx_power",
          "rssi",
          "distance",
          "mac_addr",
          "timestamp2",
      ],
      "gyroscope": ["timestamp", "x", "y", "z", "accuracy"],
      "gyroscope_uncalibrated": [
          "timestamp",
          "x",
          "y",
          "z",
          "x2",
          "y2",
          "z2",
          "accuracy",
      ],
      "magnetic_field": ["timestamp", "x", "y", "z", "accuracy"],
      "magnetic_field_uncalibrated": [
          "timestamp",
          "x",
          "y",
          "z",
          "x2",
          "y2",
          "z2",
          "accuracy",
      ],
      "rotation_vector": ["timestamp", "x", "y", "z", "accuracy"],
      "waypoint": ["timestamp", "x", "y"],
      "wifi": [
          "timestamp",
          "ssid",
          "bssid",
          "rssi",
          "frequency",
          "last_seen_timestamp",
      ],
  }

  dtype_dict = {}
  dtype_dict["accelerometer"] = {
      "timestamp": np.int64,
      "x": float,
      "y": float,
      "z": float,
      "accuracy": int,
  }
  dtype_dict["accelerometer_uncalibrated"] = {
      "timestamp": np.int64,
      "x": float,
      "y": float,
      "z": float,
      "x2": float,
      "y2": float,
      "z2": float,
      "accuracy": int,
  }
  dtype_dict["beacon"] = {
      "timestamp": np.int64,
      "uuid": str,
      "major_id": str,
      "minor_id": str,
      "tx_power": int,
      "rssi": int,
      "distance": float,
      "mac_addr": str,
      "timestamp2": np.int64,
  }
  dtype_dict["gyroscope"] = {
      "timestamp": np.int64,
      "x": float,
      "y": float,
      "z": float,
      "accuracy": int,
  }
  dtype_dict["gyroscope_uncalibrated"] = {
      "timestamp": np.int64,
      "x": float,
      "y": float,
      "z": float,
      "x2": float,
      "y2": float,
      "z2": float,
      "accuracy": int,
  }
  dtype_dict["magnetic_field"] = {
      "timestamp": np.int64,
      "x": float,
      "y": float,
      "z": float,
      "accuracy": int,
  }
  dtype_dict["magnetic_field_uncalibrated"] = {
      "timestamp": np.int64,
      "x": float,
      "y": float,
      "z": float,
      "x2": float,
      "y2": float,
      "z2": float,
      "accuracy": int,
  }
  dtype_dict["rotation_vector"] = {
      "timestamp": np.int64,
      "x": float,
      "y": float,
      "z": float,
      "accuracy": int,
  }
  dtype_dict["waypoint"] = {
      "timestamp": np.int64,
      "x": float,
      "y": float,
      "z": float
  }
  dtype_dict["wifi"] = {
      "timestamp": np.int64,
      "ssid": str,
      "bssid": str,
      "rssi": int,
      "frequency": int,
      "last_seen_timestamp": np.int64,
  }

  def _flatten(self, l):
    return list(itertools.chain.from_iterable(l))

  def multi_line_spliter(self, s):
    matches = re.finditer("TYPE_", s)
    matches_positions = [match.start() for match in matches]
    split_idx = (
        [0] +
        [matches_positions[i] - 14 for i in range(1, len(matches_positions))] +
        [len(s)])
    return [s[split_idx[i]:split_idx[i + 1]] for i in range(len(split_idx) - 1)]

  def load_df(self) -> None:
    path = Path(
        self.input_path) / "train" / self.site / self.floor / self.path_id
    path = Path(str(path).replace("_reshaped.pickle", ".txt"))

    with open(path, encoding='utf-8') as f:
      data = f.readlines()

    modified_data = []
    for s in data:
      if s.count("TYPE_") > 1:
        lines = self.multi_line_spliter(s)
        modified_data.extend(lines)
      else:
        modified_data.append(s)
    del data
    self.meta_info_len = len([d for d in modified_data if d[0] == "#"])
    self.meta_info_df = pd.DataFrame([
        m.replace("\n", "").split(":") for m in self._flatten(
            [d.split("\t") for d in modified_data if d[0] == "#"]) if m != "#"
    ])

    data_df = pd.DataFrame(
        [d.replace("\n", "").split("\t") for d in modified_data if d[0] != "#"])
    for dt in self.df_types:
      # select data type
      df_s = data_df[data_df[1] == f"TYPE_{dt.upper()}"]
      if len(df_s) == 0:
        setattr(self, dt, pd.DataFrame(columns=self.df_type_cols[dt]))
      else:
        # remove empty cols
        na_info = df_s.isna().sum(axis=0) == len(df_s)
        df_s = df_s[[i for i in na_info[na_info == False].index if i != 1
                    ]].reset_index(drop=True)

        if len(df_s.columns) != len(self.df_type_cols[dt]):
          df_s.columns = self.df_type_cols[dt][:len(df_s.columns)]
        else:
          df_s.columns = self.df_type_cols[dt]

        # set dtype
        for c in df_s.columns:
          df_s[c] = df_s[c].astype(self.dtype_dict[dt][c])

        # set DataFrame to attr
        setattr(self, dt, df_s)

  def get_site_info(self, keep_raw=False):
    self.site_info.get_site_info(keep_raw=keep_raw)

  def load_all_data(self, keep_raw=False):
    self.load_df()
    self.get_site_info(keep_raw=keep_raw)

  def __getitem__(self, item):
    if item in self.df_types:
      return getattr(self, item)
    else:
      return None

  def save(self,):
    # to be implemented
    pass


def plot_waypoint(site: str, floor: str, train_meta_df: pd.DataFrame):
  waypoint_list = []
  train_meta_s = train_meta_df.query(f"site=='{site}' and floor == '{floor}'")
  for i, r in tqdm(train_meta_s.iterrows(), total=len(train_meta_s)):
    feat = FeatureStore(site=r.site, floor=r.floor, path_id=r.path_id)
    feat.load_all_data()
    feat.waypoint["path_id"] = r.path_id
    waypoint_list.append(feat.waypoint.copy())

  waypoint_all = pd.concat(waypoint_list)

  plt.figure(
      figsize=(10, 10 * feat.site_info.site_height / feat.site_info.site_width))
  feat.site_info.show_site_image()

  for i, g in waypoint_all.groupby("path_id"):
    plt.plot(
        g.x,
        g.y,
        "-o",
        alpha=0.6,
        zorder=100,
    )

  plt.show()


def get_sensor_pos_windows(t: FeatureStore, config):
  signature_cols = config['signature_cols']
  signature_functions = config['signature_functions']
  near_waypoint_time_cutoff = config['near_waypoint_time_cutoff_s'] * 1000
  signature_half_window = config['signature_half_window_s'] * 1000
  train_sensor_frequency = config['train_sensor_frequency_s'] * 1000
  min_sensor_frequency = config['min_sensor_frequency']

  shared_time_data = t.magnetic_field["timestamp"]
  shared_time = shared_time_data.values
  waypoint_times = t.waypoint["timestamp"].values
  waypoint = t.waypoint[["x", "y"]]
  fn = t.path_id
  window_time = max(shared_time[0] + signature_half_window, waypoint_times[0])
  end_time = min(shared_time[-1] - signature_half_window, waypoint_times[-1])
  sensor_windows = []
  while window_time < end_time:
    start_sensor_time = window_time - signature_half_window
    end_sensor_time = min(window_time + signature_half_window, shared_time[-1])
    center_sensor_time = (start_sensor_time + end_sensor_time) // 2
    start_sensor_row = np.argmax(shared_time >= start_sensor_time)
    end_sensor_row = np.argmax(shared_time > end_sensor_time)
    num_sensor_obs = end_sensor_row - start_sensor_row
    window_frequency = num_sensor_obs / (end_sensor_time -
                                         start_sensor_time) * 1000

    interp_id = (window_time >= waypoint_times).sum() - 1
    assert interp_id >= 0 and interp_id < (waypoint_times.size - 1)
    if waypoint is None:
      waypoint_interp_x = None
      waypoint_interp_y = None
    else:
      frac = (window_time - waypoint_times[interp_id]) / (
          waypoint_times[interp_id + 1] - waypoint_times[interp_id])
      waypoint_interp_x = waypoint.x[interp_id] + frac * (
          waypoint.x[interp_id + 1] - waypoint.x[interp_id])
      waypoint_interp_y = waypoint.y[interp_id] + frac * (
          waypoint.y[interp_id + 1] - waypoint.y[interp_id])
    waypoint_min_time_diff = min(window_time - waypoint_times[interp_id],
                                 waypoint_times[interp_id + 1] - window_time)
    near_waypoint_t = waypoint_min_time_diff <= near_waypoint_time_cutoff

    if window_frequency >= min_sensor_frequency:
      window_summary = {
          'fn': fn,
          'start_sensor_time': start_sensor_time,
          'center_sensor_time': center_sensor_time,
          'end_sensor_time': end_sensor_time,
          'start_sensor_row': start_sensor_row,
          'end_sensor_row': end_sensor_row,
          'num_sensor_obs': num_sensor_obs,
          'window_frequency': window_frequency,
          'waypoint_interp_x': waypoint_interp_x,
          'waypoint_interp_y': waypoint_interp_y,
          'waypoint_min_time_diff': waypoint_min_time_diff,
          'near_waypoint_t': near_waypoint_t,
      }

      for c in signature_cols:
        for f_name, f in signature_functions:
          k = c + '_' + f_name
          window_summary[k] = f(
              shared_time_data[c].values[start_sensor_row:end_sensor_row])

      sensor_windows.append(window_summary)
    window_time += train_sensor_frequency

  if len(sensor_windows) == 0:
    return None
  else:
    return pd.DataFrame(sensor_windows)


def plot_magn(site: str, floor: str, train_meta_df: pd.DataFrame):
  config = {
      'signature_cols': "z",
      'signature_functions': [("max", np.max), ("mean", np.mean),
                              ("std", np.std)],
      'dist_min_std_quantile': 0.1,
      'limit_train_near_waypoints': True,
      'near_waypoint_time_cutoff_s': 1,
      'train_sensor_frequency_s': 1,
      'signature_half_window_s': 1,
      'min_sensor_frequency': 40,  # Default: 50 Hz - drop if large gaps
  }

  train_meta_s = train_meta_df.query(f"site=='{site}' and floor == '{floor}'")
  data = defaultdict(list)
  for i, r in tqdm(train_meta_s.iterrows(), total=len(train_meta_s)):
    feat = FeatureStore(site=r.site, floor=r.floor, path_id=r.path_id)
    feat.load_all_data()
    feat.waypoint["path_id"] = r.path_id
    data["waypoints"].append(feat.waypoint.copy())
    windows = get_sensor_pos_windows(feat, config)

  for k, v in data.items():
    data[k] = pd.concat(s for s in v)
  print()

  plt.figure(
      figsize=(10, 10 * feat.site_info.site_height / feat.site_info.site_width))
  feat.site_info.show_site_image()

  for i, g in waypoint_all.groupby("path_id"):
    plt.plot(
        g.x,
        g.y,
        "-o",
        alpha=0.6,
        zorder=100,
    )

  plt.show()


def floor_info(site: str, floor: str, train_meta_df: pd.DataFrame) -> None:
  train_meta_s = train_meta_df.query(f"site=='{site}' and floor == '{floor}'")
  path_id = train_meta_s.path_id.iloc[0]
  print(f"site: {site}, floor: {floor}, path_id: {path_id}")
  feature = FeatureStore(site=site, floor=floor, path_id=path_id)
  feature.load_all_data()

  # attributes of feature store
  print([c for c in dir(feature) if c[0] != "_"])

  # attributes of site info
  print([c for c in dir(feature.site_info) if c[0] != "_"])

  print(feature.site_info.site_height, feature.site_info.site_width)

  for d in feature.df_types:
    print(d)
    print(feature[d].info())
    pprint(feature[d].head())

  feature.meta_info_df.head()

  plt.figure(
      figsize=(10, 10 * feature.site_info.site_height /
               feature.site_info.site_width))
  feature.site_info.show_site_image()

  feature.site_info.draw_polygon()

  print(feature.site_info.floor_coordinates[0][:10])

  print(feature.site_info.store_coordinates[0])

  # plot_waypoint(site="5dbc1d84c1eb61796cf7c010", floor="F3", train_meta_df=train_meta)
  plot_magn(
      site="5dbc1d84c1eb61796cf7c010", floor="F3", train_meta_df=train_meta)


if __name__ == "__main__":
  # site_meta_data
  meta_path = get_data_folder() / "metadata"
  site_meta_data = pd.DataFrame([[p.split(os.sep)[-2],
                                  p.split(os.sep)[-1]]
                                 for p in glob(f"{meta_path}/**/*")])
  site_meta_data.columns = ["site_id", "floor"]
  print(site_meta_data.head())

  # train_meta_data
  train_path = get_data_folder() / "train"
  train_meta = glob(f"{train_path}/*/*/*_reshaped.pickle")
  train_meta_org = pd.DataFrame(train_meta)
  train_meta = train_meta_org[0].str.split(os.sep, expand=True)[[4, 5, 6]]
  train_meta.columns = ["site", "floor", "path_id"]
  # train_meta["path_id"] = train_meta["path_id"].str.replace(".txt", "")
  train_meta["path"] = train_meta_org[0]
  print(train_meta.head())

  # Worst for injection
  floor_info(
      site="5dbc1d84c1eb61796cf7c010", floor="F3", train_meta_df=train_meta)
