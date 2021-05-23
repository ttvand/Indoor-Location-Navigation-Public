from functools import singledispatch
from typing import Any, Dict, TypeVar, Union

import numpy as np
from shapely.geometry import GeometryCollection, LinearRing, LineString, Point, Polygon, MultiPoint, asMultiPoint
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

Geometry = TypeVar("Geometry", bound=BaseGeometry)


def maybe_to_array(geometry: Union[np.ndarray, Geometry]) -> np.ndarray:
  if isinstance(geometry, BaseGeometry):
    geometry = to_array(geometry)
  return geometry


def maybe_as_multipoint(points: Union[np.ndarray, MultiPoint]) -> MultiPoint:
  if isinstance(points, np.ndarray):
    points = asMultiPoint(points)
  return points


def maybe_to_dict(d: Union[np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
  if isinstance(d, np.ndarray):
    d = {"x": d[:, 0], "y": d[:, 1]}
  return d


@singledispatch
def to_array(geometry: Geometry) -> np.ndarray:
  """Returns a list of unique vertices of a given geometry object."""
  raise NotImplementedError(f"Unsupported Geometry {type(geometry)}")


@to_array.register
def _(geometry: Point):
  return np.array(geometry)


@to_array.register
def _(geometry: LineString):
  return np.array(geometry.coords)


@to_array.register
def _(geometry: LinearRing):
  return np.array(geometry.coords[:-1])


@to_array.register
def _(geometry: MultiPoint):
  if geometry.is_empty:
    return np.empty(shape=(0, 2))
  return np.array(geometry)


@to_array.register
def _(geometry: BaseMultipartGeometry):
  if geometry.is_empty:
    return np.empty(shape=(0, 2))
  return np.concatenate([to_array(geom) for geom in geometry])


@to_array.register
def _(geometry: Polygon):
  return to_array(GeometryCollection([geometry.exterior, *geometry.interiors]))


@singledispatch
def to_string(geometry: Geometry) -> str:
  raise NotImplementedError(f"Unsupported Geometry {type(geometry)}")


@to_string.register
def _(geometry: LinearRing):
  return ", ".join(f"({pt[0]:.3g}, {pt[1]:.3g})" for pt in geometry.coords)


@to_string.register
def _(geometry: MultiPoint):
  return ", ".join(f"({pt.x:.3g}, {pt.y:.3g})" for pt in geometry)


@to_string.register
def _(geometry: BaseMultipartGeometry):
  return str([to_string(geom) for geom in geometry])


@to_string.register
def _(geometry: LineString):
  return ", ".join(f"({pt[0]:.3g}, {pt[1]:.3g})" for pt in geometry.coords)
