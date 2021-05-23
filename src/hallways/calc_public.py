import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp2d

try:
  from utils import get_data_folder, TEST_FLOOR_MAPPING
except:
  import sys
  sys.path.append('..')
  from utils import get_data_folder, TEST_FLOOR_MAPPING

DEFAULT_WAYPOINT_PATH: Path = get_data_folder(
) / "waypoints" / "waypoint_by_hand.csv"


def generate_grid_points(site, floor, bottom_left, top_left, bottom_right,
                         top_right, N_h, N_v):
  X = np.array([0, 0, 1, 1], dtype=np.float32)
  Y = np.array([0, 1, 0, 1], dtype=np.float32)
  Zx, Zy = np.stack([bottom_left, top_left, bottom_right, top_right], axis=1)
  fx = interp2d(X, Y, Zx)
  fy = interp2d(X, Y, Zy)

  U = np.linspace(0, 1, N_h)
  V = np.linspace(0, 1, N_v)
  W = np.array(list([fx(u, v), fy(u, v)] for u, v in itertools.product(U, V)))
  W = np.squeeze(W)
  df = pd.DataFrame({
      'x': W[:, 0],
      'y': W[:, 1],
  })
  df['site'] = site
  df['floor'] = floor
  return df


def generate_individual_points(site, floor, points):
  W = np.array(points)
  df = pd.DataFrame({
      'x': W[:, 0],
      'y': W[:, 1],
  })
  df['site'] = site
  df['floor'] = floor
  return df


def generate_waypoint_by_hand():
  dfs = [
      generate_grid_points(
          site='5d2709d403f801723c32bd39',
          floor=0,
          top_left=np.array((86.4, 135.0)),
          top_right=np.array((93.4, 133.7)),
          bottom_left=np.array((83.9, 117.3)),
          bottom_right=np.array((88.7, 116.9)),
          N_h=3,
          N_v=10,
      ),
      generate_grid_points(
          site='5d2709d403f801723c32bd39',
          floor=-1,
          top_left=np.array((67, 36.5)),
          top_right=np.array((68.5, 35)),
          bottom_left=np.array((65, 32.5)),
          bottom_right=np.array((67.5, 32)),
          N_h=2,
          N_v=2,
      ),
      generate_individual_points(
          site='5da1382d4db8ce0c98bbe92e',
          floor=1,
          points=[
              (126, 160),
              (114.5, 162),
              (114.5, 156),
              (119.5, 156),
              (119.5, 147.5),
              (107, 157),
              (107.5, 145.5),
              (113.5, 145.5),
          ]),
      generate_grid_points(
          site='5d2709d403f801723c32bd39',
          floor=0,
          top_left=np.array((89, 85)),
          top_right=np.array((91, 87.5)),
          bottom_left=np.array((95, 76.5)),
          bottom_right=np.array((97.5, 77.5)),
          N_h=2,
          N_v=3,
      ),
      generate_grid_points(
          site='5d2709d403f801723c32bd39',
          floor=0,
          top_left=np.array((75.5, 92.5)),
          top_right=np.array((86, 90.5)),
          bottom_left=np.array((75, 89)),
          bottom_right=np.array((85, 87.5)),
          N_h=3,
          N_v=2,
      ),
      generate_grid_points(
          site='5d2709d403f801723c32bd39',
          floor=0,
          top_left=np.array((62.5, 86.5)),
          top_right=np.array((69.5, 91)),
          bottom_left=np.array((63.5, 85)),
          bottom_right=np.array((71, 88)),
          N_h=3,
          N_v=2,
      ),
      generate_individual_points(
          site='5d2709d403f801723c32bd39',
          floor=0,
          points=[
              (111.5, 128),
              (118.5, 123.5),
              (126.5, 116.5),
          ]),
      generate_grid_points(
          site='5da1383b4db8ce0c98bc11ab',
          floor=1,
          top_left=np.array((163, 234.5)),
          top_right=np.array((182, 235)),
          bottom_left=np.array((163, 232.5)),
          bottom_right=np.array((182, 232.5)),
          N_h=4,
          N_v=2,
      ),
      generate_individual_points(
          site='5da1382d4db8ce0c98bbe92e',
          floor=4,
          points=[
              (72.5, 155),
              (72.5, 146.5),
              (62.5, 146),
              (63.5, 148.5),
              (69, 138.5),
              (68.5, 148),
          ]),
      generate_individual_points(
          site='5da1383b4db8ce0c98bc11ab',
          floor=1,
          points=[
              (79, 216),
              (76, 220),
              (72, 215),
              (67.5, 211),
          ]),
      generate_individual_points(
          site='5da138754db8ce0c98bca82f',
          floor=2,
          points=[
              (70, 76),
              (43, 79),
              (47.5, 78.5),
              (46.5, 73),
              (42.5, 73),
              (54.5, 71.5),
              (61, 71),
          ]),
      generate_grid_points(
          site='5d27096c03f801723c31e5e0',
          floor=2,
          top_left=np.array((63.5, 59.5)),
          top_right=np.array((72, 50.5)),
          bottom_left=np.array((60.5, 57.5)),
          bottom_right=np.array((68.5, 48)),
          N_h=4,
          N_v=2,
      ),
      generate_individual_points(
          site='5d27096c03f801723c31e5e0',
          floor=2,
          points=[
              (75.5, 40.5),
              (71.5, 45),
              (57, 61.5),
          ]),
      generate_individual_points(
          site='5d27096c03f801723c31e5e0',
          floor=3,
          points=[
              (39.5, 68),
              (36.5, 65),
              (33.5, 62.5),
              (31.5, 65),
              (27.5, 61),
              (22, 56),
              (36.5, 71.5),
              (33.5, 74.5),
              (34, 79.5),
              (31, 77),
              (30.5, 83),
              (28, 80.5),
          ]),
      generate_individual_points(
          site='5d2709d403f801723c32bd39',
          floor=-1,
          points=[
              (83.5, 128),
              (88, 128.5),
              (93, 128.5),
              (97, 127.5),
              (70.5, 119.5),
          ]),
      generate_individual_points(
          site='5d2709c303f801723c3299ee',
          floor=-1,
          points=[
              (41.5, 32.5),
              (40.5, 30),
              (38, 29),
              (39.5, 27.5),
              (38, 28),
              (34.5, 28),
              (35, 30),
              (33, 28),
          ]),
      generate_individual_points(
          site='5da138754db8ce0c98bca82f',
          floor=2,
          points=[
              (75.5, 29.5),
              (79, 29),
              (83, 29),
              (83, 32.5),
              (84, 38),
              (87, 36.5),
              (92.5, 35.5),
          ]),
      generate_individual_points(
          site='5d2709d403f801723c32bd39',
          floor=-1,
          points=[
              (88, 42),
              (83, 42),
              (83.5, 46),
              (78.5, 47),
              (77, 43),
              (74.5, 48),
          ]),
      generate_grid_points(
          site='5da1383b4db8ce0c98bc11ab',
          floor=1,
          top_left=np.array((85, 177.5)),
          top_right=np.array((87.5, 175)),
          bottom_left=np.array((76.5, 163)),
          bottom_right=np.array((78.5, 161.5)),
          N_h=2,
          N_v=4,
      ),
      generate_individual_points(
          site='5d2709d403f801723c32bd39',
          floor=1,
          points=[
              (38.5, 66.5),
              (34.5, 66),
              (30.5, 65.5),
              (26.5, 65),
              (26.5, 70.5),
              (27, 75),
              (30.5, 74.5),
              (34.5, 74.5),
              (38.5, 74),
          ]),
      generate_individual_points(
          site='5da138754db8ce0c98bca82f',
          floor=1,
          points=[
              (214.5, 46),
              (220.5, 40.5),
              (222, 44.5),
              (226, 44.5),
              (226, 41),
              (232, 42),
          ]),
      generate_individual_points(
          site='5da1382d4db8ce0c98bbe92e',
          floor=4,
          points=[
              (184, 13),
              (184, 16.5),
              (186.5, 17),
              (186.5, 22.5),
              (186, 27),
              (183.5, 26.5),
              (158, 28.5),
              (180.5, 22),
              (176, 21.5),
              (171, 21.5),
              (171, 28.5),
              (171.5, 33),
              (164, 34),
              (166, 40.5),
              (163.5, 51),
              (145.5, 41.5),
              (142, 42),
              (145, 36.5),
              (142, 37),
          ]),
      generate_individual_points(
          site='5da1382d4db8ce0c98bbe92e',
          floor=2,
          points=[
              (186, 26),
              (121.5, 93.5),
              (119, 89.5),
              (118.5, 80.5),
              (125, 80),
              (133, 80),
              (136.5, 80.5),
              (136.5, 86),
          ]),
      generate_individual_points(
          site='5da1383b4db8ce0c98bc11ab',
          floor=0,
          points=[
              (86, 139),
              (84.5, 130),
          ]),
      generate_grid_points(
          site='5da958dd46f8266d0737457b',
          floor=5,
          top_left=np.array((71.5, 117)),
          top_right=np.array((78, 109)),
          bottom_left=np.array((68.5, 114.5)),
          bottom_right=np.array((75, 106.5)),
          N_h=3,
          N_v=2,
      ),
      generate_individual_points(
          site='5da138754db8ce0c98bca82f',
          floor=0,
          points=[
              (77, 54),
              (72.5, 52.5),
              (67.5, 49.5),
              (57.5, 45),
              (63, 47.5),
          ]),
      generate_individual_points(
          site='5a0546857ecc773753327266',
          floor=0,
          points=[
              (73.5, 124),
              (76, 123.5),
              (71.5, 116),
              (71, 113.5),
          ]),
      generate_individual_points(
          site='5da1383b4db8ce0c98bc11ab',
          floor=0,
          points=[
              (210, 245),
              (210, 241.5),
              (203.5, 240.5),
              (196.5, 240.5),
          ]),
      generate_individual_points(
          site='5a0546857ecc773753327266',
          floor=1,
          points=[
              (106, 77.5),
              (111, 76.5),
              (116, 76),
          ]),
      generate_individual_points(
          site='5d2709bb03f801723c32852c',
          floor=0,
          points=[
              (220.5, 39),
              (224.5, 41),
              (224, 33.5),
              (228.5, 36),
          ]),
      generate_individual_points(
          site='5da958dd46f8266d0737457b', floor=3, points=[
              (97, 44.5),
          ]),
      generate_grid_points(
          site='5da958dd46f8266d0737457b',
          floor=3,
          top_left=np.array((94, 66.5)),
          top_right=np.array((97.5, 67)),
          bottom_left=np.array((96.5, 50)),
          bottom_right=np.array((100, 50.5)),
          N_h=2,
          N_v=4,
      ),
      generate_individual_points(
          site='5dc8cea7659e181adb076a3f',
          floor=6,
          points=[
              (110, 112),
              (117, 116),
              (123, 106.5),
              (122, 112.5),
              (129, 114.5),
              (126.5, 117),
          ]),
      generate_grid_points(
          site='5da958dd46f8266d0737457b',
          floor=4,
          top_left=np.array((83.5, 160.5)),
          top_right=np.array((87, 159)),
          bottom_left=np.array((85, 151.5)),
          bottom_right=np.array((88, 151)),
          N_h=2,
          N_v=3,
      ),
      generate_individual_points(
          site='5da1383b4db8ce0c98bc11ab',
          floor=2,
          points=[
              (48, 185),
              (40.5, 177.5),
              (37.5, 181),
              (44, 181),
              (34.5, 177.5),
          ]),
      generate_individual_points(
          site='5d2709d403f801723c32bd39',
          floor=-1,
          points=[
              (108.5, 131.5),
              (103, 133),
              (83.5, 128),
              (90.5, 128.5),
              (99.5, 134),
              (100, 136),
          ]),
      generate_grid_points(
          site='5d2709d403f801723c32bd39',
          floor=-1,
          top_left=np.array((98, 130.5)),
          top_right=np.array((101, 130)),
          bottom_left=np.array((95, 119)),
          bottom_right=np.array((96.5, 118.5)),
          N_h=2,
          N_v=3,
      ),
      generate_individual_points(
          site='5d27097f03f801723c320d97',
          floor=2,
          points=[
              (60, 232.5),
              (53, 230.5),
              (54, 229),
          ]),
      generate_individual_points(
          site='5d27096c03f801723c31e5e0',
          floor=2,
          points=[
              (77, 38.5),
              (68.5, 36),
              (65, 40),
              (74.5, 41.5),
              (71.5, 45),
              (57.5, 61),
              (58, 48),
              (55.5, 45),
              (62.5, 55.5),
              (64, 53),
          ]),
      generate_individual_points(
          site='5da1382d4db8ce0c98bbe92e',
          floor=2,
          points=[
              (137, 106),
              (162, 120),
              (162, 111),
              (168.5, 106.5),
              (168.5, 111),
              (175, 111),
          ]),
      generate_individual_points(
          site='5da138b74db8ce0c98bd4774', floor=-1, points=[
              (137.5, 104.5),
          ]),
      generate_grid_points(
          site='5d27097f03f801723c320d97',
          floor=3,
          top_left=np.array((129.5, 67.5)),
          top_right=np.array((135.5, 71)),
          bottom_left=np.array((132.5, 61)),
          bottom_right=np.array((140, 65)),
          N_h=3,
          N_v=3,
      ),
      generate_individual_points(
          site='5d27097f03f801723c320d97',
          floor=3,
          points=[
              (144.5, 76),
              (146.5, 63),
              (150.5, 66.5),
              (127.5, 58.5),
          ]),
      generate_individual_points(
          site='5dbc1d84c1eb61796cf7c010',
          floor=6,
          points=[
              (116, 110),
              (112, 123.5),
          ]),
      generate_individual_points(
          site='5da138764db8ce0c98bcaa46', floor=-1, points=[
              (45, 133.5),
          ]),
      generate_individual_points(
          site='5da138764db8ce0c98bcaa46',
          floor=0,
          points=[
              (42.5, 209.5),
              (43, 204),
              (44.5, 210),
              (45, 203.5),
              (42.5, 206.5),
              (44.5, 206.5),
              (47, 206.5),
              (49, 206.5),
          ]),
      generate_grid_points(
          site='5d2709bb03f801723c32852c',
          floor=0,
          top_left=np.array((219.5, 41.5)),
          top_right=np.array((223, 44)),
          bottom_left=np.array((224, 33)),
          bottom_right=np.array((228, 36.5)),
          N_h=2,
          N_v=3,
      ),
      generate_individual_points(
          site='5da958dd46f8266d0737457b', floor=-1, points=[
              (133, 134),
          ]),
      generate_grid_points(
          site='5a0546857ecc773753327266',
          floor=2,
          top_left=np.array((104.5, 81.5)),
          top_right=np.array((116.5, 79.5)),
          bottom_left=np.array((104.5, 79.5)),
          bottom_right=np.array((116, 77)),
          N_h=4,
          N_v=2,
      ),
      generate_individual_points(
          site='5d2709b303f801723c327472',
          floor=0,
          points=[
              (201, 142),
              (201, 136),
              (202, 129.5),
              (207.5, 130.5),
              (203, 123),
              (194, 122),
              (182, 120.5),
          ]),
      generate_individual_points(
          site='5dbc1d84c1eb61796cf7c010',
          floor=4,
          points=[
              (50, 143.5),
              (47.5, 151.5),
          ]),
      generate_individual_points(
          site='5a0546857ecc773753327266',
          floor=2,
          points=[
              (84.5, 112),
              (89, 111),
          ]),
      generate_grid_points(
          site='5d2709d403f801723c32bd39',
          floor=-1,
          top_left=np.array((143, 71.5)),
          top_right=np.array((145, 71.5)),
          bottom_left=np.array((143, 61)),
          bottom_right=np.array((145, 61)),
          N_h=2,
          N_v=4,
      ),
      generate_grid_points(
          site='5d2709d403f801723c32bd39',
          floor=-1,
          top_left=np.array((143, 91)),
          top_right=np.array((145, 91)),
          bottom_left=np.array((143, 83.5)),
          bottom_right=np.array((145, 83.5)),
          N_h=2,
          N_v=3,
      ),
      generate_grid_points(
          site='5d2709d403f801723c32bd39',
          floor=-1,
          top_left=np.array((148, 91)),
          top_right=np.array((166.5, 91.5)),
          bottom_left=np.array((148, 88.5)),
          bottom_right=np.array((166.5, 88.5)),
          N_h=5,
          N_v=2,
      ),
      generate_grid_points(
          site='5d2709d403f801723c32bd39',
          floor=-1,
          top_left=np.array((178, 91.5)),
          top_right=np.array((181, 91.5)),
          bottom_left=np.array((178, 63.5)),
          bottom_right=np.array((181, 63.5)),
          N_h=2,
          N_v=6,
      ),
      generate_individual_points(
          site='5d2709d403f801723c32bd39',
          floor=-1,
          points=[
              (169, 89),
              (172.5, 89),
          ]),
      generate_grid_points(
          site='5d2709d403f801723c32bd39',
          floor=0,
          top_left=np.array((28, 119)),
          top_right=np.array((30.5, 116)),
          bottom_left=np.array((21, 110.5)),
          bottom_right=np.array((24.5, 109)),
          N_h=3,
          N_v=3,
      ),
      generate_individual_points(
          site='5dbc1d84c1eb61796cf7c010',
          floor=4,
          points=[
              (110.5, 122.5),
              (112.5, 121),
              (121.5, 107),
              (123.5, 113.5),
              (121.5, 111.5),
              (117, 110),
              (113, 113.5),
          ]),
      generate_individual_points(
          site='5d2709d403f801723c32bd39',
          floor=-1,
          points=[
              (82.5, 36.5),
              (82.5, 34.5),
              (77, 37.5),
              (76.5, 35.5),
              (70, 37.5),
              (71, 40),
              (68.5, 41),
              (68, 38.5),
          ]),
      generate_individual_points(
          site='5d27097f03f801723c320d97',
          floor=2,
          points=[
              (102, 199.5),
              (105.5, 201.5),
          ]),
      generate_individual_points(
          site='5da138764db8ce0c98bcaa46',
          floor=2,
          points=[
              (34.5, 99.5),
              (38, 100),
              (38, 103.5),
              (33, 103),
              (33, 105),
              (34.5, 108),
              (32.5, 109.5),
              (35.5, 113),
          ]),
      generate_individual_points(
          site='5da138b74db8ce0c98bd4774',
          floor=4,
          points=[
              (133, 145.5),
              (135, 148.5),
              (129.5, 148),
              (128, 146),
              (128.5, 144),
              (119.5, 153.5),
          ]),
      generate_individual_points(
          site='5dbc1d84c1eb61796cf7c010',
          floor=-1,
          points=[
              (199, 284.5),
              (182, 262.5),
              (189, 268.5),
              (186.5, 271.5),
          ]),
      generate_individual_points(
          site='5dc8cea7659e181adb076a3f',
          floor=0,
          points=[
              (253.5, 75.5),
              (253, 78.5),
              (248.5, 74),
          ]),
      generate_individual_points(
          site='5d2709bb03f801723c32852c',
          floor=1,
          points=[
              (164.5, 41.5),
              (169.5, 44.5),
              (189.5, 15),
          ]),
  ]

  df = pd.concat(dfs)
  df = df[['site', 'floor', 'x', 'y']]
  df = df.sort_values(by=['site', 'floor'])
  return df


def save_waypoints_by_hand(save_path: Optional[Path] = None) -> None:
  if save_path is None:
    save_path = DEFAULT_WAYPOINT_PATH
  save_path.parent.mkdir(parents=True, exist_ok=True)
  df = generate_waypoint_by_hand()
  df.to_csv(save_path, index=False)


def get_waypoints_by_hand(site: str,
                          floor: str,
                          source_path: Optional[Path] = None) -> np.ndarray:
  floor_int = TEST_FLOOR_MAPPING[floor]

  if source_path is None:
    source_path = DEFAULT_WAYPOINT_PATH
  df = pd.read_csv(source_path)
  waypoints = df[(df["site"] == site) & (df["floor"] == floor_int)][["x", "y"
                                                                    ]].values
  return waypoints


if __name__ == "__main__":
  save_waypoints_by_hand()
  # get_waypoints_by_hand(site="5dbc1d84c1eb61796cf7c010", floor="F3")
  # get_waypoints_by_hand(site="5d2709bb03f801723c32852c", floor="F1")
