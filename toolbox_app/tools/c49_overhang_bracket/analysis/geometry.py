from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]  # (x_ft, y_ft)

@dataclass(frozen=True)
class Segment:
    p1: Point
    p2: Point
    tag: str

    @property
    def is_vertical(self) -> bool:
        return abs(self.p2[0] - self.p1[0]) < 1e-9

    @property
    def is_horizontal(self) -> bool:
        return abs(self.p2[1] - self.p1[1]) < 1e-9

    @property
    def x_const(self) -> float:
        return self.p1[0]

    @property
    def y_min(self) -> float:
        return min(self.p1[1], self.p2[1])

    @property
    def y_max(self) -> float:
        return max(self.p1[1], self.p2[1])

def polygon_edges(poly: Sequence[Point]) -> List[Tuple[Point, Point]]:
    return list(zip(poly, poly[1:])) + [(poly[-1], poly[0])]

def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p: Point, q: Point, r: Point) -> bool:
        return (
            min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12
            and min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12
        )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True

    if abs(o1) < 1e-12 and on_seg(a1, b1, a2):
        return True
    if abs(o2) < 1e-12 and on_seg(a1, b2, a2):
        return True
    if abs(o3) < 1e-12 and on_seg(b1, a1, b2):
        return True
    if abs(o4) < 1e-12 and on_seg(b1, a2, b2):
        return True

    return False

def segment_intersects_polygon(seg1: Point, seg2: Point, poly: Sequence[Point]) -> bool:
    for e1, e2 in polygon_edges(poly):
        if segments_intersect(seg1, seg2, e1, e2):
            return True
    return False

def point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1
            if x < xinters:
                inside = not inside
    return inside

def rect_polygon(xmin: float, ymin: float, xmax: float, ymax: float) -> List[Point]:
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

def poly_intersects_poly(poly_a: Sequence[Point], poly_b: Sequence[Point]) -> bool:
    for a1, a2 in polygon_edges(poly_a):
        if segment_intersects_polygon(a1, a2, poly_b):
            return True
    if point_in_polygon(poly_a[0], poly_b):
        return True
    if point_in_polygon(poly_b[0], poly_a):
        return True
    return False
