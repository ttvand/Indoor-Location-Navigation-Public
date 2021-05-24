import numpy as np
import pandas as pd
import pathlib
import pickle
import utils
import time
from tqdm import tqdm
from shapely.geometry import LineString
from shapely.prepared import prep


def line_in_cell(line, x1, y1, x2, y2):
  if (line.coords[0][0] > x2) and (line.coords[1][0] > x2):
    return False
  if (line.coords[0][0] < x1) and (line.coords[1][0] < x1):
    return False
  if (line.coords[0][1] > y2) and (line.coords[1][1] > y2):
    return False
  if (line.coords[0][1] < y1) and (line.coords[1][1] < y1):
    return False
  #return True

  if np.isfinite(x1) and np.isfinite(x2) and np.isfinite(y1) and np.isfinite(
      y2):
    if line.intersects(LineString([(x1, y1), (x1, y2)])):
      return True
    if line.intersects(LineString([(x1, y2), (x2, y2)])):
      return True
    if line.intersects(LineString([(x2, y2), (x2, y1)])):
      return True
    if line.intersects(LineString([(x2, y1), (x1, y1)])):
      return True

    for i in range(2):
      if (line.coords[i][0] >= x1) and (line.coords[i][0] <= x2) and (
          line.coords[i][1] >= y1) and (line.coords[i][1] <= y2):
        return True
    return False
  else:
    return True


def generate_meta(walls):
  meta = {}
  for site, w0 in tqdm(walls.items()):
    if not (site in meta):
      meta[site] = {}
    for floor, w in w0.items():
      min_x = np.min([min(x.coords[0][0], x.coords[1][0]) for x in w])
      max_x = np.max([max(x.coords[0][0], x.coords[1][0]) for x in w])
      min_y = np.min([min(x.coords[0][1], x.coords[1][1]) for x in w])
      max_y = np.max([max(x.coords[0][1], x.coords[1][1]) for x in w])
      x_cuts = np.concatenate([[-np.inf],
                               np.linspace(min_x, max_x, 100), [np.inf]])
      y_cuts = np.concatenate([[-np.inf],
                               np.linspace(min_y, max_y, 100), [np.inf]])
      sets = [
          [[] for _ in range(len(y_cuts) - 1)] for _ in range(len(x_cuts) - 1)
      ]
      for idx, l in enumerate(w):
        i1 = np.searchsorted(x_cuts, l.coords[0][0]) - 1
        i2 = np.searchsorted(x_cuts, l.coords[1][0]) - 1
        j1 = np.searchsorted(y_cuts, l.coords[0][1]) - 1
        j2 = np.searchsorted(y_cuts, l.coords[1][1]) - 1
        for i in np.arange(min(i1, i2), max(i1, i2) + 1):
          for j in np.arange(min(j1, j2), max(j1, j2) + 1):
            if line_in_cell(l, x_cuts[i], y_cuts[j], x_cuts[i + 1],
                            y_cuts[j + 1]):
              sets[i][j].append(idx)

      meta[site][floor] = {
          'x_cuts': x_cuts,
          'y_cuts': y_cuts,
          'sets': sets,
      }

  repo_data_folder = pathlib.Path(__file__).parent.absolute().parent / (
    'Data files')
  with open(repo_data_folder / "walls_meta.p", 'wb') as handle:
    pickle.dump(meta, handle, protocol=4)
  return meta


class Walls():

  def __init__(self, walls_file="walls.p", meta_file="walls_meta.p"):
    #def __init__(self, walls_file="walls.p", meta_file=None):
    repo_data_folder = pathlib.Path(__file__).parent.absolute().parent / (
      'Data files')
    with open(repo_data_folder / walls_file, 'rb') as f:
      self.walls = pickle.load(f)
    if meta_file is None:
      self.meta = generate_meta(self.walls)
    else:
      with open(repo_data_folder / meta_file, 'rb') as f:
        self.meta = pickle.load(f)

  def count_walls(self, site, floor, x1, y1, x2, y2):
    # Counts number of walls between points (x1,y1) and (x2,y2) provided site id an numeric floor
    l = LineString([(x1, y1), (x2, y2)])

    #return np.sum([l.intersects(z) for z in self.walls[site][int(floor)]])

    i1 = np.searchsorted(self.meta[site][floor]['x_cuts'], l.coords[0][0]) - 1
    i2 = np.searchsorted(self.meta[site][floor]['x_cuts'], l.coords[1][0]) - 1
    j1 = np.searchsorted(self.meta[site][floor]['y_cuts'], l.coords[0][1]) - 1
    j2 = np.searchsorted(self.meta[site][floor]['y_cuts'], l.coords[1][1]) - 1

    idx = [
        self.meta[site][floor]['sets'][i][j]
        for i in np.arange(min(i1, i2),
                           max(i1, i2) + 1)
        for j in np.arange(min(j1, j2),
                           max(j1, j2) + 1)
    ]
    idx = set([item for sublist in idx for item in sublist])
    w = self.walls[site][int(floor)]
    l = prep(l)
    return np.sum([l.intersects(w[i]) for i in idx])


if __name__ == "__main__":
  preds_df = pd.read_csv(utils.get_data_folder() / '..' / 'Data files' /
                         'valid - 2021-05-02 18:29:07 assume grid.csv')
  w = Walls()

  start_time = time.time()
  res = []
  #for fn, df in tqdm(preds_df.groupby('fn')):
  for fn, df in preds_df.groupby('fn'):
    site = df['site'].values[0]
    floor = df['numeric_floor'].values[0]
    for i, seg in enumerate(
        zip(df['x_pred'].values[:-1], df['y_pred'].values[:-1],
            df['x_pred'].values[1:], df['y_pred'].values[1:])):
      res.append({
          "counter":
              w.count_walls(site, floor, seg[0], seg[1], seg[2], seg[3]),
          "err": (df['after_optim_error'].values[i] +
                  df['after_optim_error'].values[i + 1]) / 2
      })

  print(f"Time {time.time() - start_time:8.2f}s")
  res = pd.DataFrame(res)
  print(res[res['counter'] == 0]['err'].mean())
  print(res[res['counter'] > 0]['err'].mean())
  print((res['counter'] > 0).mean())
  print(len(res))

  # 78.66s
  # 0.7181569004796963
  # 1.4877503186671257
  # 0.059279665340271404
