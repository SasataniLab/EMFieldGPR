"""Module for Bounding Volume Hierarchy (BVH) implementation."""

import numpy as np
from stl import mesh

# Constants
T_TRI = 1
T_AABB = 1


class Triangle:
    """
    A class to hold information for each triangular mesh.
    """

    def __init__(self, triangle: np.ndarray):
        """
        Parameters
        ----------
        triangle : np.ndarray
            Coordinates of the triangle vertices. Expected shape is (3, 3).

        """
        if triangle.shape != (3, 3):
            msg = "Triangle must be of shape (3, 3)"
            raise ValueError(msg)

        self.triangle = triangle
        self.bbox = np.zeros((2, 3))
        self.center = np.zeros(3)

        for i in range(3):
            self.bbox[0][i] = min(triangle[:, i])
            self.bbox[1][i] = max(triangle[:, i])
            self.center[i] = (self.bbox[0][i] + self.bbox[1][i]) / 2.0


class BVHNode:
    """Single node in a Bounding Volume Hierarchy (BVH)."""

    def __init__(self):
        """Initialize BVH node."""
        self.bbox = np.zeros((2, 3))
        self.children = [-1, -1]
        self.polygons: list[Triangle] = []


class BVHBuilder:
    """Class for building Bounding Volume Hierarchy structures."""

    def __init__(self):
        """Initialize BVH builder."""
        self.nodes = [BVHNode() for _ in range(int(1e6))]
        self.used_node_count = 0

    def surface_area(self, bbox: np.ndarray) -> float:
        """Calculate surface area of bounding box."""
        dx, dy, dz = bbox[1] - bbox[0]
        return 2 * (dx * dy + dx * dz + dy * dz)

    def empty_aabb(self, bbox: np.ndarray):
        """Initialize empty axis-aligned bounding box."""
        bbox[0] = [np.inf, np.inf, np.inf]
        bbox[1] = [-np.inf, -np.inf, -np.inf]

    def merge_aabb(self, bbox1: np.ndarray, bbox2: np.ndarray, result: np.ndarray):
        """Merge two axis-aligned bounding boxes."""
        for i in range(3):
            result[0][i] = min(bbox1[0][i], bbox2[0][i])
            result[1][i] = max(bbox1[1][i], bbox2[1][i])

    def create_aabb_from_triangles(self, triangles: list[Triangle], bbox: np.ndarray):
        """Create AABB from list of triangles."""
        self.empty_aabb(bbox)
        for triangle in triangles:
            self.merge_aabb(triangle.bbox, bbox, bbox)

    def make_leaf(self, polygons: list[Triangle], node: BVHNode):
        """Create leaf node with polygons."""
        node.children = [-1, -1]
        node.polygons = polygons

    def construct_bvh_internal(self, polygons: list[Triangle], node_index: int):
        """Construct BVH internally."""
        node = self.nodes[node_index]
        self.create_aabb_from_triangles(polygons, node.bbox)

        best_cost = T_TRI * len(polygons)
        best_split_index = -1
        best_axis = -1
        sa_root = self.surface_area(node.bbox)

        for axis in range(3):
            triangles = sorted(polygons, key=lambda triangle: triangle.center[axis])

            s1 = []
            s2 = triangles.copy()
            s1_bbox = np.zeros((2, 3))
            s2_bbox = np.zeros((2, 3))
            self.empty_aabb(s1_bbox)
            self.empty_aabb(s2_bbox)

            s1_sa = [np.inf] * (len(triangles) + 1)
            s2_sa = [np.inf] * (len(triangles) + 1)

            for i in range(len(triangles) + 1):
                s1_sa[i] = abs(self.surface_area(s1_bbox))
                if s2:
                    p = s2.pop(0)
                    s1.append(p)
                    self.merge_aabb(s1_bbox, p.bbox, s1_bbox)

            for i in range(len(triangles), -1, -1):
                s2_sa[i] = abs(self.surface_area(s2_bbox))
                if s1 and s2:
                    cost = (
                        2 * T_AABB
                        + (s1_sa[i] * len(s1) + s2_sa[i] * len(s2)) * T_TRI / sa_root
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_axis = axis
                        best_split_index = i
                if s1:
                    p = s1.pop()
                    s2.insert(0, p)
                    self.merge_aabb(s2_bbox, p.bbox, s2_bbox)

        if best_axis == -1:
            self.make_leaf(polygons, node)
        else:
            triangles = sorted(
                polygons, key=lambda triangle: triangle.center[best_axis]
            )
            self.used_node_count += 1
            node.children[0] = self.used_node_count
            self.used_node_count += 1
            node.children[1] = self.used_node_count
            left = triangles[:best_split_index]
            right = triangles[best_split_index:]
            self.construct_bvh_internal(left, node.children[0])
            self.construct_bvh_internal(right, node.children[1])

    def construct_bvh(self, triangles: list[Triangle]):
        """Construct BVH from triangles."""
        self.used_node_count = 0
        self.construct_bvh_internal(triangles, 0)

    def build_bvh(self, filename: str) -> list[BVHNode]:
        """Build BVH from STL file."""
        ferrite_mesh = mesh.Mesh.from_file(filename)
        polygons = [Triangle(triangle=t) for t in ferrite_mesh.vectors]
        self.construct_bvh(polygons)
        return self.nodes[: self.used_node_count + 1]


def BVH(filename: str) -> list[BVHNode]:
    """Create BVH from STL file."""
    builder = BVHBuilder()
    return builder.build_bvh(filename)
