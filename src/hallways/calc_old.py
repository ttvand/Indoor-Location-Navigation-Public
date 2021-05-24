import json
import math
from functools import reduce
from math import pi
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import shapely
import shapely.geometry
import shapely.ops
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
    # "5da138314db8ce0c98bbf3a0",
    "5da138274db8ce0c98bbd3d2",
    "5d2709d403f801723c32bd39",
    "5d2709bb03f801723c32852c",
    "5da1382d4db8ce0c98bbe92e",
    "5a0546857ecc773753327266",
    # "5d2709a003f801723c3251bf",
    # "5d27096c03f801723c31e5e0",
    # "5d2709c303f801723c3299ee",
    # "5d27099f03f801723c32511d",
    # "5da138274db8ce0c98bbd3d2",
    # "5d27075f03f801723c2e360f",
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

np.seterr(all="raise")

mode = ("all", "train")[1]

meta_path = utils.get_data_folder() / "metadata"
site_dirs = [(meta_path / s) for s in sites]

output_folder = utils.get_data_folder() / "generated_waypoints"
plot = True


def maybe_median(array: np.ndarray, default: float = 999) -> float:
  if len(array) > 1:
    return np.median(array)
  return default


def maybe_concat(arrays: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
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


def azimuth(pt1: Point, pt2: Point) -> float:
  angle = np.arctan2(pt2.x - pt1.x, pt2.y - pt1.y)
  return angle if angle >= 0 else angle + 2 * pi


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
) -> Tuple[Point, List[Point]]:
  wall_pt = center + unit_vector * from_wall_dist
  line = LineString((wall_pt, center + unit_vector * 30))

  intersect = line.intersection(walls)

  if intersect.is_empty:
    n_points = 2
    dist = inner_dist
  else:
    wall_to_wall = center.distance(intersect) - from_wall_dist
    # n_points = int((wall_to_wall * 0.65) / inner_dist)
    n_points = round(wall_to_wall / inner_dist)
    if n_points > 0:
      dist = wall_to_wall / n_points
      n_points = math.ceil(n_points / 2)
  if n_points > 0:
    pts = [center + (unit_vector * dist * (i + 1)) for i in range(n_points)]
  else:
    pts = []

  return Point(wall_pt), pts


def wall_inner_outer_linear(
    line,
    walls: BaseGeometry,
    start,
    inner_dist,
    angle_support_dist,
    from_wall_dist,
) -> Tuple[List[Point], List[Point]]:
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
  )
  pts.extend(inner_pts1)

  wall_pt2, inner_pts2 = gen_wall_to_wall_linear(
      center,
      unit_vector=unit_vector2,
      walls=walls,
      inner_dist=inner_dist,
      from_wall_dist=from_wall_dist,
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


def near_wall_stats(
    polygons: Union[Polygon, MultiPolygon],
    known_wall_points: np.ndarray,
    known_waypoints: np.ndarray,
    min_len: float,
    maybe_wall_dist: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  assert isinstance(polygons, (Polygon, MultiPolygon, GeometryCollection, list))
  if isinstance(polygons, (GeometryCollection, list)):
    polygons = [poly for poly in polygons if isinstance(poly, Polygon)]
    not_poly = [poly for poly in polygons if not isinstance(poly, Polygon)]
    assert all(item.length < 2.0
               for item in not_poly), f"{[type(item) for item in not_poly]}"
  if isinstance(polygons, Polygon):
    polygons = [polygons]

  known_waypoints_mp = MultiPoint(known_waypoints)
  known_wallpoints_mp = MultiPoint(known_wall_points)

  rings: List[LinearRing] = [
      interior for poly in polygons for interior in poly.interiors
  ]
  rings.extend(poly.exterior for poly in polygons)

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


def along_wall(
    polygons: Union[Polygon, MultiPolygon],
    walls: GeometryCollection,
    known_wall_points: np.ndarray,
    known_waypoints: np.ndarray,
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
    generate_corner_waypoints: bool = True,
    plot: bool = False,
    min_item_count: int = 4,
    max_dist_to_consider_local: float = 40.0,
) -> Tuple[MultiPoint, MultiPoint, MultiPoint]:
  assert isinstance(polygons, (Polygon, MultiPolygon, GeometryCollection))
  if isinstance(polygons, GeometryCollection):
    polygons = [poly for poly in polygons if isinstance(poly, Polygon)]
    not_poly = [poly for poly in polygons if not isinstance(poly, Polygon)]
    assert all(item.length < 2.0
               for item in not_poly), f"{[type(item) for item in not_poly]}"
  if isinstance(polygons, Polygon):
    polygons = [polygons]

  known_waypoints_mp = MultiPoint(known_waypoints)
  known_wallpoints_mp = MultiPoint(known_wall_points)

  rings: List[LinearRing] = [
      interior for poly in polygons for interior in poly.interiors
  ]
  rings.extend(poly.exterior for poly in polygons)

  corner_points = []
  wall_points = []
  inner_points = []

  for r in rings:
    if r.length < min_len:
      continue

    if generate_corner_waypoints:
      corner_arr = locate_corners_linear(
          r, slack_lower=slack_lower, slack_upper=slack_upper)
      corner_mp = MultiPoint(corner_arr)
    else:
      corner_mp = MultiPoint(np.zeros(shape=(0, 2)))

    distance_to_wall = np.array([pt.distance(r) for pt in known_wallpoints_mp])
    wall_mask = distance_to_wall < consider_wall_dist
    current_wall_points = known_wall_points[wall_mask]
    current_known_wallpoints_mp = MultiPoint(current_wall_points)

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
      typical_wall_pt_dist = wall_pts_dist
    else:
      typical_wall_pt_dist = np.median(pt_dists[mask])

    gen_dist = wall_point_distance_multiplier * typical_wall_pt_dist
    gen_mask = pt_dists > 2 * gen_dist
    gen_inds = np.arange(len(linear_dists))[:-1][gen_mask]

    wn_sz = 4

    linear_corner_dists = [r.project(pt) for pt in corner_mp]
    corner_inds = np.searchsorted(linear_dists, linear_corner_dists)
    corner_to_wall_dists = []
    for i, ind in enumerate(corner_inds):
      local_dist_to_wall = to_wall_dist
      if len(wall_dists) > min_item_count:
        local_wall_dists = window_circular(wall_dists, ind - wn_sz,
                                           ind + 1 + wn_sz)
        local_linear_dists = window_circular(linear_dists, ind - wn_sz,
                                             ind + 1 + wn_sz)
        mask = (
            abs(local_linear_dists - linear_corner_dists[i]) <
            max_dist_to_consider_local)
        local_wall_dists = local_wall_dists[mask]
        if len(local_wall_dists) > min_item_count:
          local_dist_to_wall = np.median(local_wall_dists)

      corner_to_wall_dists.append(local_dist_to_wall)
    corner_to_wall_dists = np.array(corner_to_wall_dists)

    for start, w_dist in zip(linear_corner_dists, corner_to_wall_dists):
      corner_pts, inner_pts = wall_inner_outer_linear(
          line=r,
          walls=walls,
          start=start,
          inner_dist=typical_inner_pt_dist,
          angle_support_dist=angle_support_dist * 0.67,
          from_wall_dist=w_dist,
      )
      corner_points.extend(corner_pts)
      inner_points.extend(inner_pts)

    if len(gen_inds) > 0:
      gen_locs: List[np.ndarray] = []
      to_wall_dists: List[np.ndarray] = []

      for i in gen_inds:
        local_dist_to_wall = to_wall_dist
        if len(wall_dists) > min_item_count:
          local_wall_dists = window_circular(wall_dists, i - wn_sz,
                                             i + 1 + wn_sz)
          local_linear_dists = window_circular(linear_dists, i - wn_sz,
                                               i + 1 + wn_sz)
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
        gen_locs.append(tmp)
      to_wall_dists = np.concatenate(to_wall_dists)
      gen_locs = np.concatenate(gen_locs)

      if plot:
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
      for start, w_dist in zip(gen_locs, to_wall_dists):
        if (not corner_mp.is_empty and
            r.interpolate(start).distance(corner_mp) < gen_dist):
          continue
        wall_pts, inner_pts = wall_inner_outer_linear(
            line=r,
            walls=walls,
            start=start,
            inner_dist=typical_inner_pt_dist,
            angle_support_dist=angle_support_dist,
            from_wall_dist=w_dist,
        )
        wall_points.extend(wall_pts)
        inner_points.extend(inner_pts)

  corner_points = MultiPoint(corner_points)
  inner_points = MultiPoint(inner_points)
  wall_points = MultiPoint(wall_points)
  return corner_points, wall_points, inner_points


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

  return inner_floor, inner_clean, floor_poly, floor_plan


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
                          threshold: float) -> List[np.ndarray]:
  model = AgglomerativeClustering(
      distance_threshold=threshold, n_clusters=None, linkage="average")
  labels = model.fit_predict(points)
  clusters = [points[labels == i] for i in range(labels.max() + 1)]
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


def generate_waypoints_3(
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
    max_inner_points: int = 999,
    generate_inner_waypoints: bool = True,
    generate_corner_waypoints: bool = True,
    generate_edge_waypoints: bool = True,
    wall_point_distance_multiplier: float = 0.35,
    inner_point_distance_multiplier: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
  del generate_inner_waypoints
  del generate_corner_waypoints
  del generate_edge_waypoints
  del max_inner_points

  known_waypoints = fuse_neighbors(known_waypoints, threshold=0.8)
  known_waypoints = asMultiPoint(known_waypoints)
  known_waypoints = unary_union(known_waypoints)

  inner_floor, inner_clean, floor_poly, _ = create_floor_polygon(
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

  gen_wall_pts_dist = median_between_wall_pts_dist * wall_point_distance_multiplier
  gen_inner_pts_dist = global_median_distance[
      0] * inner_point_distance_multiplier
  corner_waypoints, wall_waypoints, inner_waypoints = along_wall(
      inner_clean,
      walls=walls,
      known_wall_points=known_wall_points,
      known_waypoints=known_waypoints,
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
  )

  corner_waypoints = unary_union(MultiPoint(corner_waypoints))
  corner_waypoints = filter_inside(corner_waypoints, inner_clean.buffer(0.05))
  corner_waypoints = fuse_neighbors(
      corner_waypoints, threshold=gen_wall_pts_dist * 0.3)

  wall_waypoints = unary_union(MultiPoint(wall_waypoints))
  wall_waypoints = filter_inside(wall_waypoints, inner_clean.buffer(0.05))
  wall_waypoints = fuse_neighbors(
      wall_waypoints, threshold=gen_wall_pts_dist * 0.6)

  inner_waypoints = unary_union(MultiPoint(inner_waypoints))
  inner_waypoints = filter_inside(inner_waypoints, inner_clean.buffer(0.05))

  generated_inner_ratio = len(inner_waypoints) / (len(wall_waypoints) + 1e-9)
  known_inner_ratio = len(not_wall_points) / len(known_wall_points)
  if generated_inner_ratio > (known_inner_ratio * 6) or known_inner_ratio < 0.1:
    inner_waypoints = np.empty(shape=(0, 2))

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
      min_distance_to_known=gen_wall_pts_dist,
      max_distance_to_known=math.inf,
  )

  wall_waypoints = filter_dist_waypoints(
      points=wall_waypoints,
      known_waypoints=known_waypoints,
      min_distance_to_known=min_distance_to_known,
      max_distance_to_known=max_distance_to_known,
  )

  corner_waypoints = maybe_to_array(corner_waypoints)
  wall_waypoints = np.concatenate((corner_waypoints, wall_waypoints))

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
                                median_near_wall_to_rest * 0.9),
      max_distance_to_known=max_distance_to_known,
  )

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


def load_waypoints(site: str, floor: str):
  data_path = utils.get_data_folder() / "train" / site / floor
  data_path = data_path.glob("*_reshaped.pickle")

  waypoints = []
  for path in data_path:
    waypoint_df = utils.load(path)["waypoint"]
    waypoints.append(waypoint_df[["x_waypoint", "y_waypoint"]])
  waypoints = np.concatenate(waypoints)

  return waypoints


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
            & (known_waypoints["floor"] == TEST_FLOOR_MAPPING[floor])]
        known_waypoints = known_waypoints[["x_waypoint", "y_waypoint"]].values
      else:
        raise AssertionError

      # wall_waypoints, inner_waypoints = generate_waypoints_2(
      #     site, floor, known_waypoints=known_waypoints, min_distance_to_known=3.0
      # )
      wall_waypoints, inner_waypoints = generate_waypoints_3(
          site,
          floor,
          known_waypoints=known_waypoints,
          min_distance_to_known=3.0)
      visualize_floor(
          site,
          floor,
          waypoints=known_waypoints,
          added_waypoints=wall_waypoints,
          title="wall waypoints",
      )
      # visualize_floor(
      #     site,
      #     floor,
      #     waypoints=known_waypoints,
      #     added_waypoints=inner_waypoints,
      #     title="inner waypoints",
      # )
      # added_waypoints_hand = get_waypoints_by_hand(site, floor)
      # visualize_floor(
      #     site,
      #     floor,
      #     waypoints=known_waypoints,
      #     added_waypoints=added_waypoints_hand,
      #     title="aiko",
      # )


if __name__ == "__main__":
  main()
  # waypoint_analysis()
