import json
import math
import pickle
import pprint
from functools import reduce
from math import pi
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

import numpy as np
import shapely
import shapely.geometry
import shapely.ops
from matplotlib import pyplot as plt
from more_itertools import pairwise
from shapely.affinity import translate
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    asMultiPoint,
    shape,
    MultiLineString,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, transform, unary_union
from shapely.prepared import prep
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

try:
  from Logic import utils
  from Logic.hallways.common import (
    to_array,
    maybe_as_multipoint,
    maybe_to_array,
    to_string,
  )
  from Logic.hallways.figures import plot_polygon, plot_polygon_data
  from Logic.utils import TEST_FLOOR_MAPPING
except:
  import utils
  from hallways.common import (
    to_array,
    maybe_as_multipoint,
    maybe_to_array,
    to_string,
  )
  from hallways.figures import plot_polygon, plot_polygon_data
  from utils import TEST_FLOOR_MAPPING

hard_locs = [
    ("5c3c44b80379370013e0fd2b", None),
    ("5dbc1d84c1eb61796cf7c010", "F3"),
]

# sites = utils.TEST_SITES
sites = [
    "5da138314db8ce0c98bbf3a0",
    "5da138274db8ce0c98bbd3d2",
    "5d2709d403f801723c32bd39",
    "5d2709bb03f801723c32852c",
    "5da1382d4db8ce0c98bbe92e",
    "5a0546857ecc773753327266",
    "5d2709a003f801723c3251bf",
    "5d27096c03f801723c31e5e0",
    # "5d2709c303f801723c3299ee",
    # "5d27099f03f801723c32511d",
    # "5da138274db8ce0c98bbd3d2",
    # "5d27075f03f801723c2e360f",
    # "5d27097f03f801723c320d97",
]
# sites = ["5d2709d403f801723c32bd39", "5d2709bb03f801723c32852c"]
# sites = ["5d2709d403f801723c32bd39"]
# sites = ["5da1382d4db8ce0c98bbe92e"]
# sites = ["5d27075f03f801723c2e360f"]
# floors = None
# floors = {"F1", "1F"}
# floors = {"F7"}
floors = {"B1", "F1", "1F", "F2", "2F", "F4", "4F"}
# floors = {"B1"}
# floors = {"F2", "2F"}
# floors = {"F3", "3F"}
# floors = {"F4", "4F"}
# floors = {"F5", "5F"}

# np.seterr(all="raise")

mode = ("all", "train")[1]

meta_path = utils.get_data_folder() / "metadata"
site_dirs = [(meta_path / s) for s in sites]

output_folder = utils.get_data_folder() / "generated_waypoints"
plot = True


def maybe_median(array: np.ndarray, default: float = 999) -> float:
  if len(array) > 1:
    return np.median(array)
  return default


def maybe_concat(arrays: Union[Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
  if isinstance(arrays, np.ndarray):
    return arrays
  if len(arrays) == 0:
    return np.zeros(shape=(0, 2))
  return np.concatenate(arrays)


# See: https://www.kaggle.com/c/indoor-location-navigation/discussion/230558
OVERRIDE_SIZES = {
    "5d27075f03f801723c2e360f": {
        "B1": {
            "height": 240.0
        },
        "F1": {
            "height": 211.0
        },
        "F2": {
            "height": 211.0
        },
        "F3": {
            "height": 211.0
        },
        "F4": {
            "height": 211.0
        },
        "F5": {
            "height": 135.3
        },
        "F6": {
            "height": 135.3
        },
        "F7": {
            "height": 135.3
        },
    }
}


def gen_wall_to_wall(center, walls, unit_vector, inner_dist):
  line = LineString((center + unit_vector * 0.05, center + unit_vector * 30))
  intersect = line.intersection(walls)
  if intersect.is_empty:
    n_points = 2
    dist = inner_dist
  else:
    wall_to_wall = center.distance(intersect)
    # n_points = int((wall_to_wall * 0.65) / inner_dist)
    n_points = round(wall_to_wall / inner_dist)
    if n_points > 0:
      dist = wall_to_wall / n_points

  if n_points > 0:
    pts = [center + (unit_vector * dist * (i + 1)) for i in range(n_points)]
  else:
    pts = []
  return pts


def wall_inner_outer(
    line: LineString,
    walls: BaseGeometry,
    st: float,
    inner_dist: float,
    angle_support_dist: float,
) -> Tuple[Point, List[Point]]:
  center = line.interpolate(st)
  left_support = line.interpolate(st - angle_support_dist)
  right_support = line.interpolate(st + angle_support_dist)

  rotated = shapely.affinity.rotate(
      LineString([left_support, right_support]), 90, origin="centroid")
  vector = shapely.affinity.scale(
      rotated,
      xfact=(1 / rotated.length),
      yfact=(1 / rotated.length),
  )
  vector = np.array(vector)
  unit_vector1 = vector[-1] - vector[0]
  unit_vector2 = vector[0] - vector[-1]

  pts = []
  pts.extend(
      gen_wall_to_wall(
          center, unit_vector=unit_vector1, walls=walls, inner_dist=inner_dist))
  pts.extend(
      gen_wall_to_wall(
          center, unit_vector=unit_vector2, walls=walls, inner_dist=inner_dist))

  return center, pts


def get_starts(known_wall_points, line, step, min_len):
  if line.length < min_len:
    return None

  ts = 0.0
  # TODO: maybe optimize
  if known_wall_points is not None and len(known_wall_points) > 0:
    known_wall_points = asMultiPoint(known_wall_points)
    try_starts = np.arange(0.0, line.length, step / 4)
    found_start = False
    for ts in try_starts:
      if line.interpolate(ts).distance(known_wall_points) > step:
        found_start = True
        break
    if not found_start:
      return None

  starts = np.arange(ts, line.length, step)
  return starts


def generate_at_points(
    ring: LinearRing,
    gen_linear_dists: np.ndarray,
    known_linear_dists: np.ndarray,
    known_wall_dists: np.ndarray,
    walls: BaseGeometry,
    default_to_wall_dist: float,
    default_inner_pt_dist: float,
    angle_support_dist: float,
    min_item_count: int,
    gen_dist: float,
    max_dist_to_consider_local: float,
    circular_window_half_size: int,
    gen_inner: bool,
    max_n_points: int,
    avoid_points: Optional[MultiPoint] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  if avoid_points is None or avoid_points.is_empty:
    avoid_points = None

  gen_inds = np.searchsorted(known_linear_dists, gen_linear_dists)
  corner_to_wall_dists = []
  for ind in gen_inds:
    local_dist_to_wall = default_to_wall_dist
    if len(known_wall_dists) > min_item_count:
      local_wall_dists = window_circular(
          known_wall_dists,
          ind - 1 - circular_window_half_size,
          ind + circular_window_half_size,
      )
      local_linear_dists = window_circular(
          known_linear_dists,
          ind - 1 - circular_window_half_size,
          ind + circular_window_half_size,
      )
      mask = (abs(local_linear_dists - known_linear_dists[ind - 1]) <
              max_dist_to_consider_local) | (
                  abs(local_linear_dists - known_linear_dists[ind]) <
                  max_dist_to_consider_local)

      local_wall_dists = local_wall_dists[mask]
      if len(local_wall_dists) > min_item_count:
        local_dist_to_wall = np.median(local_wall_dists)

    corner_to_wall_dists.append(local_dist_to_wall)
  corner_to_wall_dists = np.array(corner_to_wall_dists)

  out_cpts = []
  out_ipts = []
  for start, w_dist in zip(gen_linear_dists, corner_to_wall_dists):
    if (not avoid_points is None and
        ring.interpolate(start).distance(avoid_points) < gen_dist * 0.8):
      continue
    cpts, ipts = wall_inner_outer_linear(
        line=ring,
        walls=walls,
        start=start,
        inner_dist=default_inner_pt_dist,
        angle_support_dist=angle_support_dist * 0.67,
        from_wall_dist=w_dist,
        gen_inner=gen_inner,
        max_n_points=max_n_points,
    )
    out_cpts.extend(cpts)
    out_ipts.extend(ipts)
  return out_cpts, out_ipts


def segmentize_line(line, known_wall_points, step: float,
                    min_len: float) -> Optional[List[Point]]:
  starts = get_starts(known_wall_points, line, step, min_len)
  if starts is None:
    return None

  points = [line.interpolate(start) for start in starts]
  return points


def segmentize_gen_inner(
    line,
    walls,
    known_wall_points,
    step: float,
    inner_dist: float,
    min_len: float,
    angle_support_dist: float,
) -> Optional[Tuple[List[Point], List[Point]]]:
  starts = get_starts(known_wall_points, line, step, min_len)
  if starts is None:
    return None

  wall_pts = []
  inner_pts = []
  for st in starts:
    center, pts = wall_inner_outer(
        line,
        walls=walls,
        st=st,
        inner_dist=inner_dist,
        angle_support_dist=angle_support_dist,
    )
    wall_pts.append(center)
    inner_pts.extend(pts)

  return wall_pts, inner_pts


def segmentize_ring(
    ring: LinearRing,
    walls: BaseGeometry,
    known_wall_points,
    step: float,
    inner_dist: float,
    min_len: float,
    gen_inner: bool,
    angle_support_dist: float,
):
  lines = shapely.geometry.MultiLineString(list(pairwise(ring.coords)))
  if gen_inner:
    pts = [
        segmentize_gen_inner(
            line,
            walls=walls,
            known_wall_points=known_wall_points,
            step=step,
            inner_dist=inner_dist,
            min_len=min_len,
            angle_support_dist=angle_support_dist,
        ) for line in lines
    ]
  else:
    pts = [
        segmentize_line(
            line,
            known_wall_points,
            step,
            min_len=min_len,
        ) for line in lines
    ]

  pts = [line for line in pts if line is not None]
  return pts


def segmentize_poly(
    polygon: Polygon,
    walls: BaseGeometry,
    known_wall_points,
    step: float,
    inner_dist: float,
    min_len: float,
    gen_inner: bool,
    angle_support_dist: float,
) -> List[Tuple[float, float]]:
  rings: List[LinearRing] = [interior for interior in polygon.interiors]
  rings.append(polygon.exterior)
  coords = [
      segmentize_ring(
          r,
          walls,
          known_wall_points,
          inner_dist=inner_dist,
          step=step,
          min_len=min_len,
          gen_inner=gen_inner,
          angle_support_dist=angle_support_dist,
      ) for r in rings
  ]
  coords = sum(coords, [])
  return coords


def azimuth(pt1: Point, pt2: Point) -> float:
  angle = np.arctan2(pt2.x - pt1.x, pt2.y - pt1.y)
  return angle if angle >= 0 else angle + 2 * pi


def in_box(arr, x_min, x_max, y_min, y_max) -> np.ndarray:
  conds = (arr[:, 0] > x_min, arr[:, 0] < x_max, arr[:, 1] > y_min,
           arr[:, 1] < y_max)
  out = np.logical_and.reduce(conds)
  return out


def locate_corners(ring: LinearRing,
                   slack_lower: float = 0.15,
                   slack_upper: float = 0.15):
  min_90 = pi / 2 - slack_lower
  max_90 = pi / 2 + slack_upper
  min_270 = 3 * pi / 2 - slack_lower
  max_270 = 3 * pi / 2 + slack_upper

  line_pts = np.array(ring.coords)
  pts = [Point(pt) for pt in line_pts]
  azs = [azimuth(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
  azs = azs + [azs[0]]
  angle = np.diff(azs)
  angle[angle > 2 * pi] -= 2 * pi
  angle[angle <= 0] += 2 * pi
  mask = np.logical_or(
      np.logical_and(angle > min_90, angle < max_90),
      np.logical_and(angle > min_270, angle < max_270),
  )
  corners = line_pts[1:][mask]
  return corners


def corners_poly(polygon: Polygon, slack_lower: float,
                 slack_upper: float) -> np.ndarray:
  rings: List[LinearRing] = [interior for interior in polygon.interiors]
  rings.append(polygon.exterior)
  coords = [
      locate_corners(r, slack_lower=slack_lower, slack_upper=slack_upper)
      for r in rings
  ]
  coords = np.concatenate(coords, axis=0)
  return coords


def generate_corners(polygon: Union[Polygon, MultiPolygon], slack_lower: float,
                     slack_upper: float) -> MultiPoint:
  if isinstance(polygon, (MultiPolygon, GeometryCollection)):
    points = [
        corners_poly(poly, slack_lower=slack_lower, slack_upper=slack_upper)
        for poly in polygon
        if isinstance(poly, Polygon)
    ]
    points = np.concatenate(points, axis=0)
  else:
    points = corners_poly(
        polygon, slack_lower=slack_lower, slack_upper=slack_upper)
  points = MultiPoint(points)
  return points


def segmentize(
    polygon: Union[Polygon, MultiPolygon],
    walls: GeometryCollection,
    known_wall_points: np.ndarray,
    wall_pts_dist: float,
    inner_pts_dist: float,
    min_len: float,
    angle_support_dist: float,
    generate_inner_points: bool = False,
):
  if isinstance(polygon, MultiPolygon):
    points = [
        segmentize_poly(
            poly,
            walls=walls,
            known_wall_points=known_wall_points,
            step=wall_pts_dist,
            inner_dist=inner_pts_dist,
            min_len=min_len,
            gen_inner=generate_inner_points,
            angle_support_dist=angle_support_dist,
        ) for poly in polygon
    ]
    points = sum(points, [])
  else:
    points = segmentize_poly(
        polygon,
        walls=walls,
        known_wall_points=known_wall_points,
        step=wall_pts_dist,
        inner_dist=inner_pts_dist,
        min_len=min_len,
        gen_inner=generate_inner_points,
        angle_support_dist=angle_support_dist,
    )
  if len(points) == 0:
    return MultiPoint(), MultiPoint()

  if isinstance(points[0], list):
    wall_waypoints = MultiPoint(sum(points, []))
    return wall_waypoints, None
  else:
    wall_waypoints, inner_waypoints = zip(*points)

    wall_waypoints = [pt for pt in wall_waypoints if pt is not None]
    wall_waypoints = MultiPoint(sum(wall_waypoints, []))

    inner_waypoints = [pt for pt in inner_waypoints if pt is not None]
    inner_waypoints = MultiPoint(sum(inner_waypoints, []))
    return wall_waypoints, inner_waypoints


def window_circular(arr: np.ndarray, start: int, end: int):
  x_max = len(arr)

  if 0 <= start < x_max and 0 <= end < x_max:
    # All coordinates fit within the matrix, use underlying implementation
    return arr[start:end]

  # Need manual intervention, coordinates outside normal range of matrix
  window = np.empty((end - start), dtype=arr.dtype, order="F")
  for x_i in range(start, end):
    window[x_i - start] = arr[x_i % x_max]
  return window


def gen_wall_to_wall_linear(
    center,
    walls: BaseGeometry,
    unit_vector,
    inner_dist: float,
    from_wall_dist: float,
    gen_inner: bool,
    default_n_points: int = 3,
    max_n_points: int = 999,
) -> Tuple[np.ndarray, List[np.ndarray]]:
  wall_pt = center + unit_vector * from_wall_dist
  if not gen_inner:
    return wall_pt, []

  line = LineString((wall_pt, center + unit_vector * 30))

  intersect = line.intersection(walls)

  if intersect.is_empty:
    n_points = default_n_points
    dist = inner_dist
  else:
    wall_to_wall = center.distance(intersect) - from_wall_dist
    if wall_to_wall < 2:
      n_points = default_n_points
      dist = inner_dist
    else:
      # n_points = int((wall_to_wall * 0.65) / inner_dist)
      n_points = round(wall_to_wall / inner_dist)
      if n_points > 0:
        dist = wall_to_wall / n_points
        n_points = math.ceil(n_points / 2)
  n_points = max_n_points if n_points > max_n_points else n_points
  if n_points > 0:
    pts = [center + (unit_vector * dist * (i + 1)) for i in range(n_points)]
  else:
    pts = []

  return wall_pt, pts


def wall_inner_outer_linear(
    line,
    walls: BaseGeometry,
    start,
    inner_dist,
    angle_support_dist,
    from_wall_dist,
    gen_inner: bool,
    max_n_points: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  center = line.interpolate(start)

  left_support = line.interpolate(start - angle_support_dist)
  right_support = line.interpolate(start + angle_support_dist)

  rotated = shapely.affinity.rotate(
      LineString([left_support, right_support]), 90, origin="centroid")
  vector = shapely.affinity.scale(
      rotated,
      xfact=(1 / rotated.length),
      yfact=(1 / rotated.length),
  )
  vector = np.array(vector)
  unit_vector1 = vector[-1] - vector[0]
  unit_vector2 = vector[0] - vector[-1]

  pts = []
  wall_pt1, inner_pts1 = gen_wall_to_wall_linear(
      center,
      unit_vector=unit_vector1,
      walls=walls,
      inner_dist=inner_dist,
      from_wall_dist=from_wall_dist,
      gen_inner=gen_inner,
      max_n_points=max_n_points,
  )
  pts.extend(inner_pts1)

  wall_pt2, inner_pts2 = gen_wall_to_wall_linear(
      center,
      unit_vector=unit_vector2,
      walls=walls,
      inner_dist=inner_dist,
      from_wall_dist=from_wall_dist,
      gen_inner=gen_inner,
      max_n_points=max_n_points,
  )
  pts.extend(inner_pts2)
  wall_pts = [wall_pt1, wall_pt2]

  return wall_pts, pts


def locate_corners_linear(ring: LinearRing,
                          slack_lower: float = 0.15,
                          slack_upper: float = 0.15):
  min_90 = pi / 2 - slack_lower
  max_90 = pi / 2 + slack_upper
  min_270 = 3 * pi / 2 - slack_lower
  max_270 = 3 * pi / 2 + slack_upper

  line_pts = np.array(ring.coords)
  pts = [Point(pt) for pt in line_pts]
  azs = [azimuth(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
  azs = azs + [azs[0]]
  angle = np.diff(azs)
  angle[angle > 2 * pi] -= 2 * pi
  angle[angle <= 0] += 2 * pi
  mask = np.logical_or(
      np.logical_and(angle > min_90, angle < max_90),
      np.logical_and(angle > min_270, angle < max_270),
  )
  corners = line_pts[1:][mask]
  return corners


def get_rings(
    polygons: Union[Polygon, MultiPolygon, GeometryCollection, List[Polygon]]
) -> List[LinearRing]:
  assert isinstance(polygons, (Polygon, MultiPolygon, GeometryCollection, list))
  if isinstance(polygons, (GeometryCollection, list)):
    polygons = [poly for poly in polygons if isinstance(poly, Polygon)]
    not_poly = [poly for poly in polygons if not isinstance(poly, Polygon)]
    assert all(item.length < 2.0
               for item in not_poly), f"{[type(item) for item in not_poly]}"
  if isinstance(polygons, Polygon):
    polygons = [polygons]

  rings: List[LinearRing] = [
      interior for poly in polygons for interior in poly.interiors
  ]
  rings.extend(poly.exterior for poly in polygons)
  return rings


def near_wall_stats(
    polygons: Union[Polygon, MultiPolygon],
    known_wall_points: np.ndarray,
    known_waypoints: np.ndarray,
    min_len: float,
    maybe_wall_dist: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  rings = get_rings(polygons)

  known_waypoints_mp = MultiPoint(known_waypoints)
  known_wallpoints_mp = MultiPoint(known_wall_points)

  to_wall_distances = []
  along_wall_distances = []
  near_wall_to_rest_distances = []

  for r in rings:
    if r.length < min_len:
      continue

    distance_to_wall = np.array([pt.distance(r) for pt in known_wallpoints_mp])
    wall_mask = distance_to_wall < maybe_wall_dist
    current_wall_points = known_wall_points[wall_mask]
    current_known_wallpoints_mp = MultiPoint(current_wall_points)

    if len(current_known_wallpoints_mp) <= 1:
      continue

    rest_kps = known_waypoints_mp.difference(current_known_wallpoints_mp)

    if (len(maybe_to_array(current_known_wallpoints_mp)) == 0 or
        len(maybe_to_array(rest_kps)) == 0):
      near_dists = np.zeros(shape=(0, 2))
    else:
      near_dists = nearest_dists(current_known_wallpoints_mp, rest_kps)

    linear_dists = [r.project(pt) for pt in current_known_wallpoints_mp]
    linear_dists.append(r.length + min(linear_dists))
    linear_dists = np.array(linear_dists)
    sort_inds = np.argsort(linear_dists)
    to_wall_dists = [
        r.interpolate(wpd).distance(pt)
        for wpd, pt in zip(linear_dists, current_known_wallpoints_mp)
    ]
    to_wall_dists.append(to_wall_dists[0])
    to_wall_dists = np.array(to_wall_dists)
    linear_dists = linear_dists[sort_inds]
    to_wall_dists = to_wall_dists[sort_inds]
    wall_pts_dists = np.diff(linear_dists)

    to_wall_distances.append(to_wall_dists)
    along_wall_distances.append(wall_pts_dists)
    near_wall_to_rest_distances.append(near_dists)

  to_wall_distances = maybe_concat(to_wall_distances)
  along_wall_distances = maybe_concat(along_wall_distances)
  near_wall_to_rest_distances = maybe_concat(near_wall_to_rest_distances)

  return to_wall_distances, along_wall_distances, near_wall_to_rest_distances


def get_corners(
    polygons: Union[Polygon, MultiPolygon, GeometryCollection, List[Polygon]],
    slack_lower: float,
    slack_upper: float,
    min_len: float,
) -> np.ndarray:
  rings = get_rings(polygons)

  corner_points = []

  for r in rings:
    if r.length < min_len:
      continue

    corner_arr = locate_corners_linear(
        r, slack_lower=slack_lower, slack_upper=slack_upper)
    corner_points.append(corner_arr)
  corner_points = maybe_concat(corner_points)
  return corner_points


def filter_near_ring(points: MultiPoint, ring: LinearRing,
                     max_distance: float) -> MultiPoint:
  distance_to_ring = np.array([pt.distance(ring) for pt in points])
  points = to_array(points)
  near_ring_mask = distance_to_ring < max_distance
  near_ring_points = points[near_ring_mask]
  near_ring_points = MultiPoint(near_ring_points)
  return near_ring_points


def along_wall(
    polygons: Union[Polygon, MultiPolygon],
    walls: GeometryCollection,
    known_wall_points: np.ndarray,
    known_waypoints: np.ndarray,
    all_building_corners: np.ndarray,
    wall_pts_dist: float,
    inner_pts_dist: float,
    to_wall_dist: float,
    min_len: float,
    angle_support_dist: float,
    slack_lower: float,
    slack_upper: float,
    consider_wall_dist: float,
    wall_point_distance_multiplier: float,
    inner_point_distance_multiplier: float,
    generate_inner_waypoints: bool = True,
    generate_corner_waypoints: bool = True,
    generate_edge_waypoints: bool = True,
    plot_projections: bool = False,
    min_item_count: int = 4,
    max_dist_to_consider_local: float = 40.0,
    circular_window_half_size: int = 4,
    max_n_points: int = 999,
) -> Tuple[MultiPoint, MultiPoint, MultiPoint, MultiPoint, Dict[Tuple[int, int],
                                                                float]]:
  rings = get_rings(polygons)

  known_waypoints_mp = MultiPoint(known_waypoints)
  known_wallpoints_mp = MultiPoint(known_wall_points)
  all_building_corners_mp = MultiPoint(all_building_corners)

  gen_stats: Dict[Tuple[int, int], float] = {}
  corner_points = []
  edge_points = []
  wall_points = []
  inner_points = []

  for r in rings:
    if r.length < min_len:
      continue

    # plot_polygon(poly=Polygon(r.coords))

    if generate_corner_waypoints:
      corner_arr = locate_corners_linear(
          r, slack_lower=slack_lower, slack_upper=slack_upper)
      corner_mp = MultiPoint(corner_arr)
    else:
      corner_mp = MultiPoint(np.zeros(shape=(0, 2)))

    current_known_wallpoints_mp = filter_near_ring(
        known_wallpoints_mp, ring=r, max_distance=consider_wall_dist)
    if generate_edge_waypoints:
      edge_mp = filter_near_ring(
          all_building_corners_mp, ring=r, max_distance=consider_wall_dist)
      edge_mp = [pt for pt in edge_mp if pt.distance(corner_mp) > 1.0]
      edge_mp = MultiPoint(edge_mp)

    typical_inner_pt_dist = inner_pts_dist
    if len(current_known_wallpoints_mp) > 1:
      rest_kps = known_waypoints_mp.difference(current_known_wallpoints_mp)
      if (len(maybe_to_array(rest_kps)) >= min_item_count and
          len(maybe_to_array(current_known_wallpoints_mp)) > min_item_count):
        distances = nearest_dists(current_known_wallpoints_mp, rest_kps)
        distances = distances[(distances > 1.2) & (distances < 12.0)]
        if len(distances) > min_item_count:
          median_nearest_dist = np.median(distances)
          median_nearest_dist *= inner_point_distance_multiplier
          typical_inner_pt_dist = median_nearest_dist

      linear_dists = [r.project(pt) for pt in current_known_wallpoints_mp]
      linear_dists.append(r.length + min(linear_dists))
      linear_dists = np.array(linear_dists)
      sort_inds = np.argsort(linear_dists)
      wall_dists = [
          r.interpolate(wpd).distance(pt)
          for wpd, pt in zip(linear_dists, current_known_wallpoints_mp)
      ]
      wall_dists.append(wall_dists[0])
      wall_dists = np.array(wall_dists)
      linear_dists = linear_dists[sort_inds]
      wall_dists = wall_dists[sort_inds]
    else:
      linear_dists = np.array([0.0, r.length])
      wall_dists = np.array([to_wall_dist, to_wall_dist])

    pt_dists = np.diff(linear_dists)

    mask = (pt_dists > 1.2) & (pt_dists < 20.0)
    if len(pt_dists[mask]) < min_item_count:
      gen_dist = wall_pts_dist
    else:
      gen_dist = np.median(pt_dists[mask]) * wall_point_distance_multiplier

    gen_mask = pt_dists > 2 * gen_dist
    gen_inds = np.arange(len(linear_dists))[:-1][gen_mask]

    corner_linear_dists = np.array([r.project(pt) for pt in corner_mp])
    cpts, cripts = generate_at_points(
        ring=r,
        gen_linear_dists=corner_linear_dists,
        known_linear_dists=linear_dists,
        known_wall_dists=wall_dists,
        walls=walls,
        default_to_wall_dist=to_wall_dist,
        default_inner_pt_dist=typical_inner_pt_dist,
        angle_support_dist=angle_support_dist,
        gen_dist=gen_dist,
        min_item_count=min_item_count,
        max_dist_to_consider_local=max_dist_to_consider_local,
        circular_window_half_size=circular_window_half_size,
        gen_inner=False,
        max_n_points=max_n_points,
    )
    corner_points.extend(cpts)
    inner_points.extend(cripts)

    if generate_edge_waypoints:
      edge_linear_dists = np.array([r.project(pt) for pt in edge_mp])
      edgpts, edipts = generate_at_points(
          ring=r,
          gen_linear_dists=edge_linear_dists,
          known_linear_dists=linear_dists,
          known_wall_dists=wall_dists,
          walls=walls,
          default_to_wall_dist=to_wall_dist,
          default_inner_pt_dist=typical_inner_pt_dist,
          angle_support_dist=angle_support_dist,
          gen_dist=gen_dist,
          min_item_count=min_item_count,
          max_dist_to_consider_local=max_dist_to_consider_local,
          circular_window_half_size=circular_window_half_size,
          gen_inner=False,
          avoid_points=corner_mp,
          max_n_points=max_n_points,
      )
      edge_points.extend(edgpts)
      inner_points.extend(edipts)

    if generate_inner_waypoints:
      current_known_wallpoints_linear_dists = np.array(
          [r.project(pt) for pt in current_known_wallpoints_mp])
      _, knipts = generate_at_points(
          ring=r,
          gen_linear_dists=current_known_wallpoints_linear_dists,
          known_linear_dists=linear_dists,
          known_wall_dists=wall_dists,
          walls=walls,
          default_to_wall_dist=to_wall_dist,
          default_inner_pt_dist=typical_inner_pt_dist,
          angle_support_dist=angle_support_dist,
          gen_dist=gen_dist,
          min_item_count=min_item_count,
          max_dist_to_consider_local=max_dist_to_consider_local,
          circular_window_half_size=circular_window_half_size,
          gen_inner=generate_inner_waypoints,
          max_n_points=max_n_points,
      )
      inner_points.extend(knipts)

    if len(gen_inds) > 0:
      gen_locs: List[float] = []
      to_wall_dists: List[np.ndarray] = []
      between_dists: List[np.ndarray] = []

      for i in gen_inds:
        local_dist_to_wall = to_wall_dist
        if len(wall_dists) > min_item_count:
          local_wall_dists = window_circular(
              wall_dists,
              i - circular_window_half_size,
              i + 1 + circular_window_half_size,
          )
          local_linear_dists = window_circular(
              linear_dists,
              i - circular_window_half_size,
              i + 1 + circular_window_half_size,
          )
          mask = (abs(local_linear_dists - linear_dists[i]) <
                  max_dist_to_consider_local) | (
                      abs(local_linear_dists - linear_dists[i + 1]) <
                      max_dist_to_consider_local)
          local_wall_dists = local_wall_dists[mask]
          if len(local_wall_dists) > min_item_count:
            local_dist_to_wall = np.median(local_wall_dists)

        first = linear_dists[i]
        last = linear_dists[i + 1]
        open_dist = last - first
        n_points = round(open_dist / gen_dist)
        dist = open_dist / n_points
        tmp = np.arange(first + dist, last - dist / 2, dist)
        to_wall_dists.append(
            np.full(shape=tmp.shape, fill_value=local_dist_to_wall))
        between_dists.append(np.full(shape=tmp.shape, fill_value=dist))
        gen_locs.append(tmp)
      to_wall_dists = np.concatenate(to_wall_dists)
      gen_locs = np.concatenate(gen_locs)
      between_dists = np.concatenate(between_dists)

      if plot_projections:
        plot_polygon(Polygon(r), data_1=to_array(known_waypoints_mp))
        plot_polygon(Polygon(r), data_1=to_array(known_wallpoints_mp))
        plot_polygon(Polygon(r), data_1=to_array(current_known_wallpoints_mp))
        plot_polygon(
            Polygon(r),
            data_1=to_array(MultiPoint([r.interpolate(l) for l in gen_locs])),
        )
        plot_polygon(
            Polygon(r),
            data_1=to_array(MultiPoint([r.interpolate(l) for l in gen_locs])),
            data_2=to_array(
                MultiPoint([r.interpolate(l) for l in linear_dists])),
        )
      gen_locs[gen_locs > r.length] -= r.length
      for i, (start, w_dist) in enumerate(zip(gen_locs, to_wall_dists)):
        if (not corner_mp.is_empty and
            r.interpolate(start).distance(corner_mp) < gen_dist * 0.8):
          continue
        wall_pts, inner_pts = wall_inner_outer_linear(
            line=r,
            walls=walls,
            start=start,
            inner_dist=typical_inner_pt_dist,
            angle_support_dist=angle_support_dist,
            from_wall_dist=w_dist,
            gen_inner=generate_inner_waypoints,
            max_n_points=max_n_points,
        )
        wall_points.extend(wall_pts)
        inner_points.extend(inner_pts)
        for pt in wall_pts:
          gen_stats[(round(pt[0]), round(pt[1]))] = float(between_dists[i])
        for pt in inner_pts:
          gen_stats[(round(pt[0]), round(pt[1]))] = float(between_dists[i])

  corner_points = MultiPoint(corner_points)
  edge_points = MultiPoint(edge_points)
  inner_points = MultiPoint(inner_points)
  wall_points = MultiPoint(wall_points)
  return corner_points, edge_points, wall_points, inner_points, gen_stats


def load_json(path: Path) -> Dict[str, Any]:
  return json.load(open(path, "r"))


def fc_to_gc_json(fcd: Dict) -> Dict:
  gcd = {
      "type": "geometrycollection",
      "geometries": [f["geometry"] for f in fcd["features"]],
  }
  return gcd


def gc_to_mp(gc: GeometryCollection) -> MultiPolygon:
  polygons = [g for g in gc.geoms if g.geom_type == "Polygon"]
  res = MultiPolygon(polygons)
  if not res.is_valid:
    # Attempt to make valid by merging overlaps
    res = unary_union(res)
  assert res.is_valid
  return res


def move_inside(points: Union[MultiPoint, np.ndarray],
                geom: BaseGeometry,
                tolerance: float = 0.2) -> MultiPoint:
  points = maybe_as_multipoint(points)
  points = [
      nearest_points(pt, geom)[1] if 0.0 < pt.distance(geom) < tolerance else pt
      for pt in points
  ]
  points = MultiPoint(points)
  return points


def grid_within_polygon(
    polygon: BaseGeometry,
    grid_step: float = 2.0,
) -> MultiPoint:
  latmin, lonmin, latmax, lonmax = polygon.bounds
  x, y = np.meshgrid(
      np.arange(latmin, latmax, grid_step), np.arange(lonmin, lonmax,
                                                      grid_step))

  points = MultiPoint(list(zip(x.flatten(), y.flatten())))
  prep_polygon = prep(polygon)
  valid_points = [i for i in points if prep_polygon.contains(i)]
  valid_points = MultiPoint(valid_points)

  return valid_points


def filter_dist_waypoints(
    points: Union[np.ndarray, MultiPoint],
    known_waypoints: Union[np.ndarray, MultiPoint],
    min_distance_to_known: float = 4.0,
    max_distance_to_known: float = 20.0,
    limit_to_hull: bool = False,
    min_hull_distance: float = 0.0,
) -> MultiPoint:
  points = maybe_as_multipoint(points)
  known_waypoints = maybe_as_multipoint(known_waypoints)

  if len(known_waypoints) == 0:
    return points

  po = [
      pt for pt in points if max_distance_to_known > pt.distance(
          known_waypoints) > min_distance_to_known
  ]
  if limit_to_hull:
    known_waypoints_approx = known_waypoints.convex_hull
    po = [
        pt for pt in po
        if pt.distance(known_waypoints_approx) <= min_hull_distance
    ]
  return MultiPoint(po)


def filter_inside(points: Union[MultiPoint, np.ndarray],
                  poly: BaseGeometry) -> MultiPoint:
  if isinstance(points, np.ndarray):
    points = asMultiPoint(points)

  prep_polygon = prep(poly)
  points = [i for i in points if prep_polygon.contains(i)]
  points = MultiPoint(points)
  return points


def recursive_get(d, *keys):
  return reduce(lambda c, k: c.get(k, {}), keys, d)


def create_floor_polygon(site: str, floor: str):
  meta_folder = meta_path / site / floor
  floor_json = load_json(meta_folder / "geojson_map.json")
  floor_info = load_json(meta_folder / "floor_info.json")["map_info"]

  override_height = recursive_get(OVERRIDE_SIZES, site, floor, "height")
  if override_height:
    floor_info["height"] = override_height

  override_width = recursive_get(OVERRIDE_SIZES, site, floor, "width")
  if override_width:
    floor_info["width"] = override_width

  geometry_json = fc_to_gc_json(floor_json)

  # First polygon is the floor (no walls)
  floor_poly = shape(geometry_json["geometries"][0])
  floor_poly = unary_union(floor_poly)
  floor_poly = floor_poly.buffer(0).simplify(0.1)

  # Remove floor_poly to get only walls
  geometry_json["geometries"] = geometry_json["geometries"][1:]
  m = shape(geometry_json)

  x_offset = floor_poly.bounds[0]
  y_offset = floor_poly.bounds[1]
  x_scale = floor_info["width"] / (floor_poly.bounds[2] - floor_poly.bounds[0])
  y_scale = floor_info["height"] / (floor_poly.bounds[3] - floor_poly.bounds[1])

  def scale_fn(x, y, z=None):
    if z is None:
      return x * x_scale, y * y_scale
    else:
      return x * x_scale, y * y_scale, z

  floor_poly = translate(floor_poly, -x_offset, -y_offset)
  floor_poly = transform(scale_fn, floor_poly)

  m = translate(m, -x_offset, -y_offset)
  m = transform(scale_fn, m)

  floor_plan = gc_to_mp(m)
  floor_plan_buildings = [g for g in m.geoms if g.geom_type == "Polygon"]

  inner_floor = floor_poly.difference(floor_plan)

  # Cleanup very thin "corridors" and simplify jagged corners
  tolerance = 0.2
  distance = 0.4
  cofactor = 1.3
  inner_clean = (
      inner_floor.buffer(-distance).buffer(
          distance *
          cofactor).intersection(inner_floor).simplify(tolerance=tolerance))
  inner_clean = unary_union(inner_clean)

  return inner_floor, inner_clean, floor_poly, floor_plan_buildings


def visualize_floor(
    site: str,
    floor: str,
    waypoints: np.ndarray,
    added_waypoints: Optional[Union[Dict, np.ndarray]] = None,
    title: str = "",
):
  inner_floor, inner_clean, floor_poly, _ = create_floor_polygon(
      site=site, floor=floor)
  plot_polygon(
      inner_floor,
      bounds=floor_poly.bounds,
      title=f"{title} site: {site} floor: {floor}",
      data_1=waypoints,
      data_2=added_waypoints,
  )


def generate_waypoints(
    site: str,
    floor: str,
    known_waypoints: np.ndarray,
    dist_between_waypoints=2.0,
    min_distance_to_known=5.23,
    max_distance_to_known=8.5,
) -> np.ndarray:

  inner_floor, inner_clean, floor_poly, _ = create_floor_polygon(
      site=site,
      floor=floor,
  )

  generated_waypoints = grid_within_polygon(
      inner_clean, grid_step=dist_between_waypoints)
  generated_waypoints = filter_dist_waypoints(
      points=generated_waypoints,
      known_waypoints=known_waypoints,
      min_distance_to_known=min_distance_to_known,
      max_distance_to_known=max_distance_to_known,
  )

  if len(generated_waypoints) == 0:
    return np.empty(shape=(0, 2))

  return np.array(generated_waypoints)


def to_radians(arr: np.ndarray) -> np.array:
  arr[:, 0] = arr[:, 0] * np.pi / 180
  arr[:, 1] = arr[:, 1] * np.pi / 180
  return arr


def get_nearest(src_points,
                candidates,
                k_neighbors=5) -> Tuple[np.ndarray, np.ndarray]:
  """Find nearest neighbors for all source points from a set of candidate points"""
  src_points = src_points.reshape(-1, 2)
  candidates = candidates.reshape(-1, 2)
  neigh = NearestNeighbors(n_neighbors=k_neighbors)
  neigh.fit(candidates)
  distances, indices = neigh.kneighbors(src_points)
  return distances, indices


def knearest_self(points: np.ndarray,
                  k_neighbors: int = 5,
                  skip_self: bool = True) -> Tuple[np.ndarray, np.ndarray]:
  distances, indices = get_nearest(points, points, k_neighbors=k_neighbors)

  if skip_self:
    distances = distances[:, 1:]
    indices = indices[:, 1:]
  return distances, indices


def get_neighbor_clusters(points: np.ndarray,
                          threshold: float,
                          linkage: str = "average") -> List[np.ndarray]:
  model = AgglomerativeClustering(
      distance_threshold=threshold, n_clusters=None, linkage=linkage)
  labels = model.fit_predict(points)
  clusters = [points[labels == i] for i in range(labels.max() + 1)]
  assert len(labels) == len(np.concatenate(clusters))
  return clusters


def fuse_neighbors(points: Union[MultiPoint, np.ndarray],
                   threshold: float = 1.5) -> np.ndarray:
  points = maybe_to_array(points)
  if len(points) == 0:
    return np.zeros(shape=(0, 2))
  if len(points) == 1:
    return points

  clusters = get_neighbor_clusters(points, threshold=threshold)
  points = [
      Point(pts.reshape(-1)) if pts.shape[0] == 1 else MultiPoint(pts)
      for pts in clusters
  ]
  points = [pt.centroid for pt in points]
  points = to_array(MultiPoint(points))
  return points


def fuse_neighbors_local(
    points: Union[MultiPoint, np.ndarray],
    gen_stats: Dict[Tuple[int, int], float],
    inner_threshold_default: float,
    outer_threshold: float,
    min_item_count: int = 4,
    local_ratio: float = 0.5,
) -> np.ndarray:
  points = maybe_to_array(points)
  if len(points) == 0:
    return np.zeros(shape=(0, 2))
  if len(points) == 1:
    return points

  out = []
  outer_clusters = get_neighbor_clusters(
      points, threshold=outer_threshold, linkage="average")
  for cl in outer_clusters:
    if len(cl) > 1:
      cluster_keys = [(round(pt[0]), round(pt[1])) for pt in cl]
      stats_keys = gen_stats.keys()
      vals = [gen_stats[k] for k in cluster_keys if k in stats_keys]
      if len(vals) > min_item_count:
        lcl = np.mean(vals) * 0.95
        gbl = inner_threshold_default
        thresh = lcl * local_ratio + gbl * (1 - local_ratio)
      else:
        thresh = inner_threshold_default

      inner_clusters = get_neighbor_clusters(cl, threshold=thresh)
      points = [
          Point(pts.reshape(-1)) if pts.shape[0] == 1 else MultiPoint(pts)
          for pts in inner_clusters
      ]
      points = [pt.centroid for pt in points]
      out.extend(points)
  out = to_array(MultiPoint(out))
  return out


def local_histogram(points: Union[np.ndarray, MultiPoint],
                    k_neighbors: int = 5,
                    title: str = "") -> None:
  points = maybe_to_array(points)
  k_neighbors = min(k_neighbors, points.shape[0])
  distances, indices = knearest_self(
      points, k_neighbors=k_neighbors, skip_self=True)
  distances = distances[distances < 9]
  plt.hist(distances, bins=20)
  plt.title(title)
  plt.show()


def nearest_dists(
    points: Union[np.ndarray, MultiPoint],
    other_points: Optional[Union[np.ndarray, MultiPoint]] = None,
) -> np.ndarray:
  points = maybe_to_array(points)
  if other_points is None:
    distances, indices = knearest_self(points, k_neighbors=2, skip_self=True)
  else:
    other_points = maybe_to_array(other_points)
    distances, _ = get_nearest(points, other_points, k_neighbors=1)
  return distances


def local_stats(
    points: Union[np.ndarray, MultiPoint],
    other_points: Optional[Union[np.ndarray, MultiPoint]] = None,
    k_neighbors: int = 5,
    min_dist: float = 1.2,
    max_dist: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  points = maybe_to_array(points)
  k_neighbors = min(k_neighbors, points.shape[0])
  if other_points is None:
    distances, indices = knearest_self(
        points, k_neighbors=k_neighbors, skip_self=True)
  else:
    other_points = maybe_to_array(other_points)
    distances, indices = get_nearest(
        points, other_points, k_neighbors=k_neighbors)

  means = []
  stds = []
  medians = []
  for column in distances.T:
    column = column[np.logical_and(min_dist < column, column < max_dist)]
    if len(column) > 0:
      means.append(column.mean(axis=0))
      stds.append(column.std(axis=0))
      medians.append(np.median(column, axis=0))
  return np.array(means), np.array(stds), np.array(medians)


def generate_waypoints_4(
    site: str,
    floor: str,
    known_waypoints: np.ndarray,
    min_distance_to_known: float = 3.0,
    corner_min_distance_to_known: float = 1.05,
    max_distance_to_known: float = 30.0,
    dist_maybe_wall_pt: float = 2.0,
    dist_definitely_wall_pt: float = 0.4,
    corner_radians_slack_upper: float = (pi / 2) * 0.34,
    corner_radians_slack_lower: float = (pi / 2) * 0.34,
    angle_support_dist: float = 1.5,
    generate_inner_waypoints: bool = True,
    generate_corner_waypoints: bool = True,
    generate_edge_waypoints: bool = False,
    wall_point_distance_multiplier: float = 0.40,
    inner_point_distance_multiplier: float = 0.7,
    max_inner_points: int = 999,
) -> Tuple[np.ndarray, np.ndarray]:
  known_waypoints = np.unique(known_waypoints, axis=0)
  known_waypoints = fuse_neighbors(known_waypoints, threshold=0.8)
  known_waypoints = asMultiPoint(known_waypoints)
  known_waypoints = unary_union(known_waypoints)

  inner_floor, inner_clean, floor_poly, floor_plan = create_floor_polygon(
      site=site,
      floor=floor,
  )

  bndry = floor_poly.boundary
  if isinstance(bndry, MultiLineString):
    outer_poly = MultiPolygon([Polygon(b) for b in bndry])
  else:
    outer_poly = Polygon(floor_poly.boundary)
  outer_poly = unary_union(outer_poly)

  assert outer_poly.is_valid
  assert floor_poly.is_valid
  outer_walls = floor_poly.envelope.difference(outer_poly)
  inner_walls = floor_poly.difference(inner_clean)
  if isinstance(inner_walls, Polygon):
    inner_walls = [inner_walls]
  walls = unary_union([item.buffer(0) for item in inner_walls] +
                      [outer_walls.buffer(0)])
  walls = walls.buffer(0.05)
  assert walls.is_valid
  assert not walls.is_empty

  # plot_polygon(inner_walls, title="inner_walls")
  # plot_polygon(outer_walls, title="outer_walls")
  # plot_polygon(walls, title="walls")

  distance_to_wall = np.array([pt.distance(walls) for pt in known_waypoints])

  maybe_wall_mask = distance_to_wall < dist_maybe_wall_pt
  wall_mask = maybe_wall_mask

  if len(maybe_wall_mask) > 10:
    assert len(maybe_wall_mask[maybe_wall_mask]) / len(maybe_wall_mask) > 0.2

  consider_wall_dist = 0.0
  known_wall_points = np.empty(shape=(0, 2))
  if maybe_wall_mask.any():
    typical_distance_to_wall_global = np.quantile(
        distance_to_wall[maybe_wall_mask], q=0.7)
    consider_wall_dist = max((typical_distance_to_wall_global * 1.5),
                             dist_definitely_wall_pt)
    wall_mask = distance_to_wall < max(
        (typical_distance_to_wall_global * 1.5), dist_definitely_wall_pt)
    known_wall_points = np.array(known_waypoints)[wall_mask]

  dist_mean, dist_std, global_median_distance = local_stats(known_waypoints)

  not_wall_points = (
      np.empty(shape=(0, 2))
      if wall_mask.all() else np.array(known_waypoints)[~wall_mask])

  # plot_polygon(
  #     walls.buffer(typical_distance_to_wall_global),
  #     bounds=floor_poly.bounds,
  #     title=f"inner site: {site} floor: {floor} walls",
  # )
  # plot_polygon(
  #     inner_clean.buffer(-typical_distance_to_wall_global),
  #     bounds=floor_poly.bounds,
  #     title=f"inner site: {site} floor: {floor} inner",
  # )

  (
      to_wall_distances,
      along_wall_distances,
      near_wall_to_rest_distances,
  ) = near_wall_stats(
      polygons=inner_clean,
      known_wall_points=known_wall_points,
      known_waypoints=known_waypoints,
      min_len=2.0,
      maybe_wall_dist=dist_maybe_wall_pt,
  )
  median_to_wall_dist = maybe_median(
      to_wall_distances[to_wall_distances < consider_wall_dist])
  median_between_wall_pts_dist = maybe_median(
      along_wall_distances[(along_wall_distances > 1.1)
                           & (along_wall_distances < 20)])
  median_near_wall_to_rest = maybe_median(
      near_wall_to_rest_distances[(near_wall_to_rest_distances > 1.1)
                                  & (near_wall_to_rest_distances < 10)])

  if generate_edge_waypoints:
    all_building_corners = get_corners(
        polygons=floor_plan,
        slack_lower=corner_radians_slack_lower,
        slack_upper=corner_radians_slack_upper,
        min_len=6.0,
    )
    all_building_corners = unary_union(MultiPoint(all_building_corners))
  else:
    all_building_corners = MultiPoint(np.zeros(shape=(0, 2)))
  # plot_polygon(floor_plan, data_1=all_corners)

  gen_wall_pts_dist = median_between_wall_pts_dist * wall_point_distance_multiplier
  gen_inner_pts_dist = global_median_distance[
      0] * inner_point_distance_multiplier
  (
      corner_waypoints,
      edge_waypoints,
      wall_waypoints,
      inner_waypoints,
      gen_stats,
  ) = along_wall(
      inner_clean,
      walls=walls,
      known_wall_points=known_wall_points,
      known_waypoints=known_waypoints,
      all_building_corners=all_building_corners,
      wall_pts_dist=gen_wall_pts_dist,
      inner_pts_dist=gen_inner_pts_dist,
      to_wall_dist=median_to_wall_dist,
      min_len=gen_wall_pts_dist,
      angle_support_dist=angle_support_dist,
      slack_lower=corner_radians_slack_lower,
      slack_upper=corner_radians_slack_upper,
      consider_wall_dist=consider_wall_dist,
      wall_point_distance_multiplier=wall_point_distance_multiplier,
      inner_point_distance_multiplier=inner_point_distance_multiplier,
      generate_inner_waypoints=generate_inner_waypoints,
      generate_edge_waypoints=generate_edge_waypoints,
      generate_corner_waypoints=generate_corner_waypoints,
      max_n_points=max_inner_points,
  )
  edge_waypoints = filter_inside(edge_waypoints, inner_clean.buffer(0.1))

  inner_waypoints = unary_union(MultiPoint(inner_waypoints))
  inner_waypoints = filter_inside(inner_waypoints, inner_clean.buffer(0.05))

  wall_waypoints = unary_union(MultiPoint(wall_waypoints))
  wall_waypoints = filter_inside(wall_waypoints, inner_clean.buffer(0.05))
  wall_waypoints = fuse_neighbors_local(
      wall_waypoints,
      gen_stats=gen_stats,
      outer_threshold=50,
      inner_threshold_default=gen_wall_pts_dist * 0.6,
  )
  if len(edge_waypoints) > 0:
    dists = nearest_dists(edge_waypoints, known_waypoints)
    mask = dists < dist_maybe_wall_pt
    dists_near = dists[mask]
    building_corner_ratio = len(dists_near) / len(dists) + 1e-9

    if building_corner_ratio > 0.35:
      wall_waypoints = MultiPoint(wall_waypoints)
      wall_waypoints = [
          wall_pt if wall_pt.distance(edge_waypoints) > gen_wall_pts_dist * 0.4
          else nearest_points(wall_pt, edge_waypoints)[1]
          for wall_pt in wall_waypoints
      ]
      wall_waypoints = MultiPoint(wall_waypoints)

  wall_waypoints = fuse_neighbors(
      wall_waypoints, threshold=gen_wall_pts_dist * 0.3)

  corner_waypoints = unary_union(MultiPoint(corner_waypoints))
  corner_waypoints = filter_inside(corner_waypoints, inner_clean.buffer(0.05))
  corner_waypoints = fuse_neighbors(
      corner_waypoints, threshold=gen_wall_pts_dist * 0.3)

  inner_waypoints = fuse_neighbors_local(
      inner_waypoints,
      gen_stats=gen_stats,
      outer_threshold=60,
      inner_threshold_default=min(gen_inner_pts_dist * 0.95,
                                  gen_wall_pts_dist * 0.95),
      local_ratio=0.4,
  )
  inner_waypoints = fuse_neighbors(
      inner_waypoints,
      threshold=min(gen_inner_pts_dist * 0.95, gen_wall_pts_dist * 0.95),
  )

  corner_waypoints = filter_dist_waypoints(
      points=corner_waypoints,
      known_waypoints=known_waypoints,
      min_distance_to_known=corner_min_distance_to_known,
      max_distance_to_known=max_distance_to_known,
  )
  wall_waypoints = filter_dist_waypoints(
      points=wall_waypoints,
      known_waypoints=corner_waypoints,
      min_distance_to_known=gen_wall_pts_dist * 0.50,
      max_distance_to_known=math.inf,
  )

  wall_waypoints = filter_dist_waypoints(
      points=wall_waypoints,
      known_waypoints=known_waypoints,
      min_distance_to_known=min_distance_to_known,
      max_distance_to_known=max_distance_to_known,
  )

  wall_waypoints = np.concatenate(
    (maybe_to_array(corner_waypoints),
     maybe_to_array(wall_waypoints)))

  inner_waypoints = filter_dist_waypoints(
      points=inner_waypoints,
      known_waypoints=wall_waypoints,
      min_distance_to_known=gen_inner_pts_dist * 0.95,
      max_distance_to_known=math.inf,
  )
  inner_waypoints = filter_dist_waypoints(
      points=inner_waypoints,
      known_waypoints=known_waypoints,
      min_distance_to_known=max(min_distance_to_known,
                                median_near_wall_to_rest * 0.9),
      max_distance_to_known=max_distance_to_known,
  )

  generated_inner_ratio = len(inner_waypoints) / (len(wall_waypoints) + 1e-9)
  known_inner_ratio = len(not_wall_points) / len(known_wall_points)
  if generated_inner_ratio > (known_inner_ratio *
                              4.0) or known_inner_ratio < 0.1:
    inner_waypoints = np.empty(shape=(0, 2))

  wall_waypoints = move_inside(
      wall_waypoints, inner_clean.buffer(0.02), tolerance=0.2)
  wall_waypoints = filter_inside(wall_waypoints, inner_clean.buffer(0.02))
  inner_waypoints = filter_inside(inner_waypoints, inner_clean.buffer(0.02))

  wall_waypoints = maybe_to_array(wall_waypoints)
  inner_waypoints = maybe_to_array(inner_waypoints)

  if wall_waypoints.ndim != 2:
    raise ValueError(
        f"Unexpected shape at output. Wall waypoints shape: {wall_waypoints.shape}"
    )

  if inner_waypoints.ndim != 2:
    raise ValueError(
        f"Unexpected shape at output. Inner waypoints shape: {inner_waypoints.shape}"
    )

  # assert len(wall_waypoints) > (len(known_waypoints) * 0.1), f"{site} {floor}"

  return wall_waypoints, inner_waypoints


def generate_waypoints_2(
    site: str,
    floor: str,
    known_waypoints: np.ndarray,
    min_distance_to_known: float = 3.0,
    corner_min_distance_to_known: float = 1.05,
    max_distance_to_known: float = 30.0,
    dist_maybe_wall_pt: float = 1.4,
    dist_definitely_wall_pt: float = 0.4,
    corner_radians_slack_upper: float = (pi / 2) * 0.34,
    corner_radians_slack_lower: float = (pi / 2) * 0.34,
    angle_support_dist: float = 1.5,
    generate_inner_waypoints: bool = True,
    wall_point_distance_multiplier: float = 0.4,
    inner_point_distance_multiplier: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
  known_waypoints = asMultiPoint(known_waypoints)
  known_waypoints = unary_union(known_waypoints)

  inner_floor, inner_clean, floor_poly, _ = create_floor_polygon(
      site=site,
      floor=floor,
  )
  floor_boundary = floor_poly.boundary
  inner_walls = floor_poly.difference(inner_clean)
  if isinstance(inner_walls, Polygon):
    inner_walls = [inner_walls]
  walls = unary_union([item.buffer(0) for item in inner_walls] +
                      [floor_boundary.buffer(0)])
  distance_to_wall = np.array([pt.distance(walls) for pt in known_waypoints])

  maybe_wall_mask = distance_to_wall < dist_maybe_wall_pt
  wall_mask = maybe_wall_mask

  typical_distance_to_wall_global = 9999
  known_wall_points = np.empty(shape=(0, 2))
  if maybe_wall_mask.any():
    typical_distance_to_wall_global = np.quantile(
        distance_to_wall[maybe_wall_mask], q=0.4)
    wall_mask = distance_to_wall < max(
        (typical_distance_to_wall_global * 1.5), dist_definitely_wall_pt)
    known_wall_points = np.array(known_waypoints)[wall_mask]

  dist_mean, dist_std, dist_median = local_stats(known_waypoints)
  # local_histogram(points=known_waypoints, k_neighbors=2, title=f"{site} {floor}")
  dist_mean_wall_pts, dist_std_wall_pts, dist_median_wall_pts = local_stats(
      known_wall_points, k_neighbors=3)

  typical_distance_global = dist_median[0]
  typical_distance_wall_pts_global = dist_median_wall_pts[0]

  not_wall_points = (
      np.empty(shape=(0, 2))
      if wall_mask.all() else np.array(known_waypoints)[~wall_mask])

  try:
    (
        dist_mean_to_wall_pts,
        dist_std_to_wall_pts,
        dist_median_to_wall_pts,
    ) = local_stats(
        known_wall_points,
        other_points=not_wall_points,
        k_neighbors=2,
        max_dist=8.0)
    typical_distance_to_wall_pts_global = dist_median_to_wall_pts[0]
  except (ValueError, IndexError):
    typical_distance_to_wall_pts_global = 9999

  # TODO: User some sort of local stats instead of just the global ones
  #  Maybe cluster_local_stats below
  #  Not sure how to assign new areas for insertion of points to
  #  the corresponding local stats.
  #  Can use HDBSCAN predict on candidate points from global stats and then adjust to local
  #  or just vanilla euclidean distance to cluster
  #
  # model_hdb = HDBSCAN(
  #     min_cluster_size=5,
  #     min_samples=2,
  #     cluster_selection_epsilon=float(dist_median[0] / 2),
  #     prediction_data=True,
  # )
  # labels_hdb = model_hdb.fit_predict(known_waypoints)
  # plot_polygon_data(
  #     inner_floor,
  #     bounds=inner_floor.bounds,
  #     points=known_waypoints,
  #     color_labels=labels_hdb,
  #     title="Clustering hdbscan",
  # )
  # clusters_hbd = [
  #     np.array(known_waypoints)[labels_hdb == i] for i in range(labels_hdb.max())
  # ]
  # cluster_local_stats: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
  # for c in clusters_hbd:
  #     dist_mean, dist_std, dist_median = local_stats(c)
  #     cluster_local_stats.append((dist_mean, dist_std, dist_median))

  # plot_polygon(inner_clean, bounds=floor_poly.bounds, data_1=known_wall_points)

  corner_waypoints = generate_corners(
      inner_clean.buffer(-typical_distance_to_wall_global * 0.8).simplify(0.4),
      slack_lower=corner_radians_slack_lower,
      slack_upper=corner_radians_slack_upper,
  )
  corner_waypoints = MultiPoint(corner_waypoints)
  corner_waypoints = to_array(unary_union(corner_waypoints))
  corner_waypoints = filter_dist_waypoints(
      points=corner_waypoints,
      known_waypoints=known_waypoints,
      min_distance_to_known=corner_min_distance_to_known,
      max_distance_to_known=max_distance_to_known,
  )

  # plot_polygon(
  #     walls.buffer(typical_distance_to_wall_global),
  #     bounds=floor_poly.bounds,
  #     title=f"inner site: {site} floor: {floor} walls",
  # )
  # plot_polygon(
  #     inner_clean.buffer(-typical_distance_to_wall_global),
  #     bounds=floor_poly.bounds,
  #     title=f"inner site: {site} floor: {floor} inner",
  # )

  gen_wall_pts_dist = (
      typical_distance_wall_pts_global * wall_point_distance_multiplier)
  gen_inner_pts_dist = (
      typical_distance_to_wall_pts_global * inner_point_distance_multiplier)
  wall_waypoints, inner_waypoints = segmentize(
      inner_clean.buffer(-typical_distance_to_wall_global),
      walls=walls.buffer(typical_distance_to_wall_global),
      known_wall_points=known_wall_points,
      wall_pts_dist=gen_wall_pts_dist,
      inner_pts_dist=gen_inner_pts_dist,
      min_len=gen_wall_pts_dist,
      generate_inner_points=generate_inner_waypoints,
      angle_support_dist=angle_support_dist,
  )

  wall_waypoints = to_array(unary_union(wall_waypoints))
  wall_waypoints = filter_dist_waypoints(
      points=wall_waypoints,
      known_waypoints=known_waypoints,
      min_distance_to_known=gen_wall_pts_dist,
      max_distance_to_known=max_distance_to_known,
  )
  wall_waypoints = filter_dist_waypoints(
      points=wall_waypoints,
      known_waypoints=corner_waypoints,
      min_distance_to_known=gen_wall_pts_dist,
      max_distance_to_known=math.inf,
  )

  # labels, strengths = hdbscan.approximate_predict(model_hdb, to_array(wall_waypoints))

  # model_agg = AgglomerativeClustering(distance_threshold=2.0, n_clusters=None)
  # labels = model_agg.fit_predict(known_waypoints)
  # plot_polygon_data(inner_floor, bounds=inner_floor.bounds, points=known_waypoints, color_labels=labels, title="Clustering agglomerative")

  wall_waypoints = fuse_neighbors(
      wall_waypoints, threshold=gen_wall_pts_dist * 0.8)
  known_waypoints = to_array(known_waypoints)

  wall_waypoints = move_inside(wall_waypoints, inner_floor)
  wall_waypoints = filter_dist_waypoints(
      points=wall_waypoints,
      known_waypoints=known_waypoints,
      min_distance_to_known=min_distance_to_known,
      max_distance_to_known=max_distance_to_known,
  )

  if inner_waypoints is not None and len(inner_waypoints) > 0:
    inner_waypoints = filter_inside(
        inner_waypoints, inner_clean.buffer(-typical_distance_to_wall_global))
    if len(inner_waypoints) == 0:
      inner_waypoints = np.empty(shape=(0, 2))
    else:
      inner_waypoints = to_array(unary_union(inner_waypoints))
      inner_waypoints = fuse_neighbors(
          inner_waypoints,
          threshold=min(gen_inner_pts_dist * 0.95, gen_wall_pts_dist * 0.95),
      )
    inner_waypoints = maybe_to_array(inner_waypoints).reshape(-1, 2)
  else:
    inner_waypoints = np.empty(shape=(0, 2))

  generated_inner_ratio = len(inner_waypoints) / (len(wall_waypoints) + 1e-9)
  known_inner_ratio = len(not_wall_points) / len(known_wall_points)
  if generated_inner_ratio > (known_inner_ratio * 6) or known_inner_ratio < 0.1:
    inner_waypoints = np.empty(shape=(0, 2))

  corner_waypoints = np.array(corner_waypoints)
  corner_waypoints = corner_waypoints.reshape(-1, 2)
  wall_waypoints = np.array(wall_waypoints)
  wall_waypoints = wall_waypoints.reshape(-1, 2)

  wall_waypoints = np.concatenate((wall_waypoints, corner_waypoints))

  wall_waypoints = unary_union(MultiPoint(wall_waypoints))
  wall_waypoints = filter_inside(wall_waypoints, inner_clean.buffer(0.1))

  inner_waypoints = unary_union(MultiPoint(inner_waypoints))
  inner_waypoints = filter_inside(inner_waypoints, inner_clean.buffer(0.1))

  inner_waypoints = filter_dist_waypoints(
      points=inner_waypoints,
      known_waypoints=wall_waypoints,
      min_distance_to_known=gen_inner_pts_dist * 0.50,
      max_distance_to_known=math.inf,
  )
  inner_waypoints = filter_dist_waypoints(
      points=inner_waypoints,
      known_waypoints=known_waypoints,
      min_distance_to_known=max(min_distance_to_known,
                                typical_distance_to_wall_pts_global * 0.9),
      max_distance_to_known=max_distance_to_known,
  )

  wall_waypoints = to_array(wall_waypoints)
  inner_waypoints = to_array(inner_waypoints)

  return wall_waypoints, inner_waypoints


def load_waypoints(site: str, floor: str):
  data_path = utils.get_data_folder() / "train" / site / floor
  data_path = data_path.glob("*_reshaped.pickle")

  waypoints = []
  for path in data_path:
    waypoint_df = utils.load(path)["waypoint"]
    waypoints.append(waypoint_df[["x_waypoint", "y_waypoint"]])
  waypoints = np.concatenate(waypoints)

  return waypoints


def waypoint_analysis(load_floor: bool = False):
  dist_metrics = {}
  for site_dir in site_dirs:
    site = site_dir.stem
    dist_metrics[site] = {}
    for floor_dir in site_dir.iterdir():
      floor = floor_dir.stem
      dist_metrics[site][floor] = {}
      if floors and floor_dir.stem not in floors:
        continue

      waypoints = load_waypoints(site, floor)
      multi_waypoints = MultiPoint(list(zip(waypoints["x"], waypoints["y"])))
      multi_waypoints = unary_union(multi_waypoints)  # deduplicate

      dists = []
      for i, pt in enumerate(multi_waypoints):
        dists.append(
            min(
                pt.distance(multi_waypoints[:i]),
                pt.distance(multi_waypoints[i + 1:]),
            ))
      dists = np.array(dists)

      dist_metrics[site][floor]["mean"] = dists.mean()
      dist_metrics[site][floor]["mean"] = dists.mean()
      dist_metrics[site][floor]["std"] = dists.std()
      dist_metrics[site][floor]["9q"] = np.quantile(dists, 0.9)

      if load_floor:
        inner_floor, inner_clean, floor_poly, floor_plan = create_floor_polygon(
            site=site,
            floor=floor,
        )
      pprint.pprint(dist_metrics[site][floor])

  pprint.pprint(dist_metrics)
  pickle.dump(
      dist_metrics,
      open(utils.get_data_folder() / "dist_metrics.pickle", "wb"),
      protocol=pickle.HIGHEST_PROTOCOL,
  )


def main():
  for site_dir in site_dirs:
    for floor_dir in site_dir.iterdir():
      if floors and floor_dir.stem not in floors:
        continue

      site = site_dir.stem
      floor = floor_dir.stem

      # inner_floor, inner_clean, floor_poly, floor_plan = create_floor_polygon(
      #     site=site,
      #     floor=floor,
      # )

      if mode == "all":
        known_waypoints = load_waypoints(site, floor)
      elif mode == "train":
        waypoints_path = utils.get_data_folder() / "train_waypoints_timed.csv"
        known_waypoints = pd.read_csv(waypoints_path)
        known_waypoints = known_waypoints.loc[
            (known_waypoints["site_id"] == site)
            & (known_waypoints["floor"] == TEST_FLOOR_MAPPING[floor])
            & (known_waypoints["mode"] == "train")]
        known_waypoints = known_waypoints[["x_waypoint", "y_waypoint"]].values
      else:
        raise AssertionError

      wall_waypoints, inner_waypoints = generate_waypoints_4(
          site,
          floor,
          known_waypoints=known_waypoints,
          generate_inner_waypoints=True,
          generate_edge_waypoints=True,
      )
      all_waypoints = maybe_concat((wall_waypoints, inner_waypoints))
      # visualize_floor(
      #     site,
      #     floor,
      #     waypoints=known_waypoints,
      #     added_waypoints=wall_waypoints,
      #     title="wall waypoints",
      # )
      # visualize_floor(
      #     site,
      #     floor,
      #     waypoints=known_waypoints,
      #     added_waypoints=inner_waypoints,
      #     title="inner waypoints",
      # )
      visualize_floor(
          site,
          floor,
          waypoints=known_waypoints,
          added_waypoints=all_waypoints,
          title="all waypoints",
      )


if __name__ == "__main__":
  main()
  # waypoint_analysis()
