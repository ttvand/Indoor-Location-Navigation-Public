from functools import partial
from math import sqrt
from typing import Union, Optional, Tuple, Dict, List

from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
from descartes import PolygonPatch
from matplotlib import pyplot as plt
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, MultiPoint

from hallways.common import maybe_to_dict, maybe_to_array

plt.rcParams["figure.figsize"] = [16, 9]

GM = (sqrt(5) - 1.0) / 2.0
W = 8.0
H = W * GM
SIZE = (W, H)

BLUE = "#6699cc"
GRAY = "#999999"
DARKGRAY = "#333333"
YELLOW = "#ffcc33"
GREEN = "#339933"
RED = "#ff3333"
BLACK = "#000000"

COLOR_ISVALID = {
    True: BLUE,
    False: RED,
}


def plot_line(ax, ob, color=GRAY, zorder=1, linewidth=3, alpha=1):
  x, y = ob.xy
  ax.plot(
      x,
      y,
      color=color,
      linewidth=linewidth,
      solid_capstyle="round",
      zorder=zorder,
      alpha=alpha,
  )


def plot_coords(ax, ob, color=GRAY, zorder=1, alpha=1):
  x, y = ob.xy
  ax.plot(x, y, "o", color=color, zorder=zorder, alpha=alpha)


def color_isvalid(ob, valid=BLUE, invalid=RED):
  if ob.is_valid:
    return valid
  else:
    return invalid


def color_issimple(ob, simple=BLUE, complex=YELLOW):
  if ob.is_simple:
    return simple
  else:
    return complex


def plot_line_isvalid(ax, ob, **kwargs):
  kwargs["color"] = color_isvalid(ob)
  plot_line(ax, ob, **kwargs)


def plot_line_issimple(ax, ob, **kwargs):
  kwargs["color"] = color_issimple(ob)
  plot_line(ax, ob, **kwargs)


def plot_bounds(ax, ob, zorder=1, alpha=1):
  x, y = zip(*list((p.x, p.y) for p in ob.boundary))
  ax.plot(x, y, "o", color=BLACK, zorder=zorder, alpha=alpha)


def add_origin(ax, geom, origin):
  x, y = xy = affinity.interpret_origin(geom, origin, 2)
  ax.plot(x, y, "o", color=GRAY, zorder=1)
  ax.annotate(
      str(xy), xy=xy, ha="center", textcoords="offset points", xytext=(0, 8))


def set_limits(ax, x0, xN, y0, yN):
  ax.set_xlim(x0, xN)
  # ax.set_xticks(range(x0, xN+1))
  ax.set_ylim(y0, yN)
  # ax.set_yticks(range(y0, yN+1))
  ax.set_aspect("equal")


def plot_polygon(
    poly: Union[Polygon, MultiPolygon, List[Polygon]],
    bounds: Optional[Tuple] = None,
    title: str = "",
    data_1: Optional[np.ndarray] = None,
    data_2: Optional[np.ndarray] = None,
    plot_floor: bool = True,
) -> None:
  fig, ax = plt.subplots(dpi=120)

  if plot_floor:
    if isinstance(poly, list):
      for p in poly:
        path = PolygonPatch(p, fc=BLUE, ec=BLUE, alpha=0.5)
        ax.add_patch(path)
    else:
      path = PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.5)
      ax.add_patch(path)

  scatter_1 = None
  if data_1 is not None:
    scatter_1 = ax.scatter(
        x=data_1[:, 0], y=data_1[:, 1], marker="+", alpha=0.7, s=4 * 2, c="red")

  scatter_2 = None
  if data_2 is not None:
    scatter_2 = ax.scatter(
        x=data_2[:, 0],
        y=data_2[:, 1],
        marker="+",
        alpha=0.7,
        s=4 * 2,
        c="blue")

  annot = ax.annotate(
      "",
      xy=(0, 0),
      xytext=(15, 15),
      textcoords="offset points",
      bbox=dict(boxstyle="round", fc="w"),
      arrowprops=dict(arrowstyle="->"),
  )
  annot.set_visible(False)

  def update_annot(ind, sc):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = f"{pos[0]:.4}, {pos[1]:.4}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

  def hover(event, sc):
    vis = annot.get_visible()
    if event.inaxes == ax:
      cont, ind = sc.contains(event)
      if cont:
        update_annot(ind, sc)
        annot.set_visible(True)
        fig.canvas.draw_idle()
      else:
        if vis:
          annot.set_visible(False)
          fig.canvas.draw_idle()

  fig.canvas.mpl_connect("motion_notify_event", partial(hover, sc=scatter_2))

  if title:
    ax.set_title(title)

  if bounds:
    bounds = tuple(int(p) for p in bounds)
    set_limits(ax, bounds[0], bounds[2], bounds[1], bounds[3])
  else:
    ax.autoscale_view()
  plt.show()
  # plt.savefig()


def plot_polygon_data(
    poly: Union[Polygon, MultiPolygon],
    points: Union[MultiPoint, np.ndarray],
    color_labels: np.ndarray,
    bounds: Optional[Tuple] = None,
    title: str = "",
) -> None:
  points = maybe_to_array(points)

  fig, ax = plt.subplots(dpi=120)

  path = PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.4)
  ax.add_patch(path)

  gray = "#E6E6E3"
  palette = sns.husl_palette(n_colors=max(color_labels)).as_hex()
  if min(color_labels) == -1:
    palette.append(gray)
    color_labels[color_labels == -1] = max(color_labels) + 1
  cmap = ListedColormap(palette)

  ax.scatter(
      x=points[:, 0],
      y=points[:, 1],
      alpha=0.8,
      s=4 * 2,
      c=color_labels,
      cmap=cmap)

  if title:
    ax.set_title(title)

  if bounds:
    bounds = tuple(int(p) for p in bounds)
    set_limits(ax, bounds[0], bounds[2], bounds[1], bounds[3])
  else:
    ax.autoscale_view()
  plt.show()
