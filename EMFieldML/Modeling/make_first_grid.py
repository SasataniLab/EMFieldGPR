"""Functions for creating Level-1 of Aligned Edge Polycube mesh and Adaptive Exterior Grid."""

import sys

import numpy as np

from EMFieldML.config import config
from EMFieldML.Modeling.calculate_plane_point_distance import (
    make_plane_from_edge,
    search_near_points_from_plane,
)
from EMFieldML.Modeling.default_config import default_config
from EMFieldML.Modeling.RayIntersect import (
    calculate_point_ray_hit,
    calculate_point_ray_hit_all,
)


class Point:
    """
    Class for points that divide the surface of the ferrite shield into a mesh.

    Attributes
    ----------
    xyz : ndarray(3)
        Coordinates of the point.
    normal : ndarray(3)
        Vector indicating the direction the point will move next.
    corner : bool
        Whether the point is a corner of the shape or not.

    """

    def __init__(self, point):
        """
        Parameters
        ----------
        point : ndarray(3)
            Coordinates of the point.

        """
        self.xyz = np.zeros(3)
        self.normal = np.zeros(3)
        self.corner = 0
        for i in range(3):
            self.xyz[i] = point[i]


class Point_outside:
    """
    Class for predicted points on the outside of the ferrite shield.

    Attributes
    ----------
    xyz : ndarray(3)
        Coordinates of the point.
    index : int
        If the point is the same as a point on the surface of the ferrite shield, the index of that point in 'points'.
        Default is -1.
    mode : int
        Where this point belongs. 0 : predicted point, 1 : TX, 2 : RX.
    default_point : ndarray(3)
        Coordinates of the point before it moves (used for calculating weights, etc. when the shape of the ferrite changes).
    weight : ndarray(n)
        Weight for performing linear blend skinning on the points of the ferrite shield.
    parent : list(int,)
        Index of the parent points.
    edges : list(int,)
        Store the index of the edge extending from this point to simplify the search.
    plane : list(int,)
        Store the index of the plane to which this point belongs to simplify the search.

    """

    def __init__(self, point):
        """
        Parameters
        ----------
        point : ndarray(3)
            Coordinates of the point.

        """
        self.xyz = np.zeros(3)
        self.index = -1
        self.mode = 0  # 0: 外部、1:TX、2:RX
        for i in range(3):
            self.xyz[i] = point[i]
        self.default_point = np.copy(point)
        self.weight = self.calculate_weight(point)
        self.parent = []
        self.edges = []
        self.planes = []
        self.affilication = -1

    def calculate_weight(self, point):
        """
        Calculate the distance from the given point to the points of the initial shape of the ferrite shield and convert it to weight.

        Parameters
        ----------
        point : ndarray(3)
            Coordinates of the point.

        Returns
        -------
        np.array(all_d) : ndarray(n)
            Weight used for shape transformation.

        """
        # フェライトシールドのデフォルトの点の位置
        points_default = default_config.points_default

        weight_sum = 0
        min_limit = 1e-8

        all_d = []
        # まずはall_dには点と点の距離を入れていく。weight_sumには距離の1.5乗に反比例するweightを足していき、全体の合計を求める
        for i in range(36):
            d = np.linalg.norm(points_default[i] - point, ord=2)
            all_d.append(d)
            if d < min_limit:
                weight_sum = np.inf
            else:
                weight_sum += 1 / d ** (1.5)
        for i in range(36):
            d = np.linalg.norm(
                np.array([points_default[i][0], points_default[i][1], -0.06]) - point,
                ord=2,
            )
            all_d.append(d)
            if d < min_limit:
                weight_sum = np.inf
            else:
                weight_sum += 1 / d ** (1.5)
        for i in range(36):
            d = np.linalg.norm(
                np.array([points_default[i][0], points_default[i][1], 0.05]) - point,
                ord=2,
            )
            all_d.append(d)
            if d < min_limit:
                weight_sum = np.inf
            else:
                weight_sum += 1 / d ** (1.5)
        for i in range(36):
            d = np.linalg.norm(
                np.array([points_default[i][0], points_default[i][1], 0.11]) - point,
                ord=2,
            )
            all_d.append(d)
            if d < min_limit:
                weight_sum = np.inf
            else:
                weight_sum += 1 / d ** (1.5)

        # all_dに重みを足していく。ただし、weight_sumによって正規化をする
        for i in range(144):
            if all_d[i] < min_limit:
                all_d[i] = 1
            else:
                all_d[i] = (1 / all_d[i] ** (1.5)) / weight_sum

        return np.array(all_d)

    def calculate_nextpoint(self, next_point):
        """Calculate next point position."""
        points_default = default_config.points_default

        changes = []
        for i in range(36):
            change = next_point[i] - points_default[i]
            changes.append(change)
        for i in range(36):
            change = next_point[36 + i] - np.array(
                [points_default[i][0], points_default[i][1], -0.06]
            )
            changes.append(change)
        for i in range(36):
            change = next_point[72 + i] - np.array(
                [points_default[i][0], points_default[i][1], 0.05]
            )
            changes.append(change)
        for i in range(36):
            change = next_point[108 + i] - np.array(
                [points_default[i][0], points_default[i][1], 0.11]
            )
            changes.append(change)
        changes = np.array(changes)
        self.xyz = np.copy(self.default_point)
        for i in range(144):
            self.xyz += changes[i] * self.weight[i]


class Edge:
    """Edge class for mesh generation."""

    def __init__(self, points, point1_index, point2_index):
        """Initialize edge between two points."""
        self.node = [point1_index, point2_index]
        self.normal = points[point1_index].normal + points[point2_index].normal
        self.children = [-1, -1]

        vector = points[point2_index].xyz - points[point1_index].xyz
        vector /= np.linalg.norm(vector, ord=2)
        if np.linalg.norm(self.normal, ord=2) < 1e-7:
            print(
                point1_index,
                point2_index,
                points[point1_index].normal,
                points[point2_index].xyz,
            )
            sys.exit(["make invalid edge."])
        self.calculate_ray(vector)

    def calculate_ray(self, vector):
        """Calculate ray intersection with edge."""
        s = -(
            vector[0] * self.normal[0]
            + vector[1] * self.normal[1]
            + vector[2] * self.normal[2]
        ) / (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
        ray_dir = s * vector + self.normal
        ray_dir = ray_dir / np.linalg.norm(ray_dir, ord=2)
        self.normal = ray_dir


class Edge_outside:
    """Edge class for outside points."""

    def __init__(self, index1, index2):
        """Initialize outside edge between two points."""
        self.node = [index1, index2]
        self.children = [-1, -1]
        self.center = -1


class Plane:
    """Plane class for mesh generation."""

    def __init__(self, points, edges, points_index):
        """Initialize plane with four points."""
        self.node = [points_index[0], points_index[1], points_index[2], points_index[3]]
        self.edge = [-1, -1, -1, -1]
        for i in range(4):
            j = (i + 1) % 4
            for k in range(len(edges)):
                if (
                    edges[k].node[0] == points_index[i]
                    and edges[k].node[1] == points_index[j]
                ):
                    self.edge[i] = k
                if (
                    edges[k].node[0] == points_index[j]
                    and edges[k].node[1] == points_index[i]
                ):
                    self.edge[i] = k
            if self.edge[i] == -1:
                print(points_index)
                sys.exit(["make invalid plane."])
        self.normal = (
            points[points_index[0]].normal
            + points[points_index[1]].normal
            + points[points_index[2]].normal
            + points[points_index[3]].normal
        )
        self.children = [-1, -1, -1, -1]
        if np.linalg.norm(self.normal, ord=2) < 1e-7:
            sys.exit(["make invalid plane."])
        self.normal = self.normal / np.linalg.norm(self.normal, ord=2)


class Plane_outside:
    """Plane class for outside points."""

    def __init__(self, index1, index2, index3, index4):
        """Initialize outside plane with four points."""
        self.node = [index1, index2, index3, index4]
        self.edge = [-1, -1, -1, -1]
        self.children = [-1, -1, -1, -1]
        self.center = -1


class Box:
    """Box class for mesh generation."""

    def __init__(self, points_index):
        """Initialize box with eight corner points."""
        self.node = [
            points_index[0],
            points_index[1],
            points_index[2],
            points_index[3],
            points_index[4],
            points_index[5],
            points_index[6],
            points_index[7],
        ]
        self.children = [-1 for _ in range(8)]
        self.edge_node = []
        self.plane_node = []


def initial_point(nowpoints, model_points, i, A, B, C, normal):
    """
    Project 8 corner points of a cube onto the shape.
    Create a plane using adjacent corner points and project to the nearest point on the plane.

    Parameters
    ----------
    nowpoints : list(ndarray(3))
        投影する前の角の8点の座標のリスト
    model_points : list(ndarray(3))
        形状の点の座標のみを抽出したリスト
    i : int
        対応するnow_pointsのindex
    A : int
        隣接する角の点のindex
    B : int
        隣接する角の点のindex
    C : int
        隣接する角の点のindex
    normal : ndarray(3)
        その点における法線ベクトル。角の点なので隣接する平面の法線ベクトルを合計したもの。

    Returns
    -------
    class Point
        投影後の点

    """
    edgeA = nowpoints[A]
    edgeB = nowpoints[i] + (nowpoints[B] - nowpoints[i]) * 10
    edgeC = nowpoints[i] + (nowpoints[C] - nowpoints[i]) * 10
    a, b, c, d = make_plane_from_edge(nowpoints[i], edgeA, edgeB, edgeC)
    ans_point = search_near_points_from_plane(model_points, a, b, c, d)
    point = Point(ans_point[1])
    point.normal = np.array(normal) / np.linalg.norm(np.array(normal))
    point.corner = 1
    return point


def initial_edge(nodes, initial_point, vector1, vector2):
    """Create initial edge with ray intersection."""
    ray_dir = (vector2 - vector1) / np.linalg.norm(vector2 - vector1, ord=2)
    t_min = 0
    t_max = 0
    for i in range(100):
        t = i * 0.05
        ray_org = t * vector1 + initial_point
        hit = calculate_point_ray_hit_all(nodes, ray_org, ray_dir)
        if len(hit) == 0:
            t_min = t
        else:
            t_max = t
            break
    if t_max == 0:
        print(t_min, t_max, ray_dir, ray_org)
        sys.exit("There are any error in initial edge making process.")
    while True:
        t = (t_max + t_min) / 2
        ray_org = t * vector1 + initial_point
        hit = calculate_point_ray_hit_all(nodes, ray_org, ray_dir)
        if len(hit) == 0:
            t_min = t
        elif len(hit) == 1:
            break
        elif len(hit) == 2:
            if np.linalg.norm(hit[0] - hit[1], ord=2) < 1e-6:
                break
            t_max = t
        else:
            t_max = t
        if t_max - t_min <= 1e-9:
            sys.exit("error")
    return hit[0]


def calculate_ray(vector, normal):
    """Calculate ray direction from vector and normal."""
    s = -(vector[0] * normal[0] + vector[1] * normal[1] + vector[2] * normal[2]) / (
        vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2
    )
    ray_dir = s * vector + normal
    return ray_dir / np.linalg.norm(ray_dir, ord=2)


def initial_edge_point(nodes, initial_point, vector1, vector2, normal):
    """Create initial edge point with normal."""
    point = Point(initial_edge(nodes, initial_point, vector1, vector2))
    point.normal = np.array(normal) / np.linalg.norm(np.array(normal))
    point.corner = 1
    return point


def initial_plane_point(nodes, point, ray, normal):
    """Create initial plane point with normal."""
    plane_point = Point(calculate_point_ray_hit(nodes, point, ray)[1])
    plane_point.normal = np.array(normal) / np.linalg.norm(np.array(normal))
    return plane_point


def define_initial_normal(points, center, neighbor):
    """Define initial normal vector for points."""
    direction = np.empty((0, 3), float)
    for i in range(len(neighbor)):
        direction = np.append(
            direction,
            [
                (points[neighbor[i]].xyz - points[center].xyz)
                / np.linalg.norm(points[neighbor[i]].xyz - points[center].xyz)
            ],
            axis=0,
        )
    normal = np.array([0.0, 0.0, 0.0])
    for i in range(len(neighbor)):
        normal += np.cross(direction[i], direction[(i + 1) % len(neighbor)])
    normal /= np.linalg.norm(normal)
    return normal


def calculate_weight(initial):
    """Calculate weights for linear blend skinning."""
    points_default = np.array(
        [
            [-0.25, -0.25 + 0.50 * 5 / 5],
            [-0.25, -0.25 + 0.50 * 4 / 5],
            [-0.25, -0.25 + 0.50 * 3 / 5],
            [-0.25, -0.25 + 0.50 * 2 / 5],
            [-0.25, -0.25 + 0.50 * 1 / 5],
            [-0.25 + 0.50 * 0 / 5, -0.25],
            [-0.25 + 0.50 * 1 / 5, -0.25],
            [-0.25 + 0.50 * 2 / 5, -0.25],
            [-0.25 + 0.50 * 3 / 5, -0.25],
            [-0.25 + 0.50 * 4 / 5, -0.25],
            [0.25, -0.25 + 0.50 * 0 / 5],
            [0.25, -0.25 + 0.50 * 1 / 5],
            [0.25, -0.25 + 0.50 * 2 / 5],
            [0.25, -0.25 + 0.50 * 3 / 5],
            [0.25, -0.25 + 0.50 * 4 / 5],
            [-0.25 + 0.50 * 5 / 5, 0.25],
            [-0.25 + 0.50 * 4 / 5, 0.25],
            [-0.25 + 0.50 * 3 / 5, 0.25],
            [-0.25 + 0.50 * 2 / 5, 0.25],
            [-0.25 + 0.50 * 1 / 5, 0.25],
        ]
    )
    weights = []
    total_sum = 0  #
    for i in range(20):
        v_norm_i = (points_default[(i - 1) % 20] - initial) / np.linalg.norm(
            points_default[(i - 1) % 20] - initial, ord=2
        )
        v_norm_ii = (points_default[i] - initial) / np.linalg.norm(
            points_default[i] - initial, ord=2
        )
        v_norm_iii = (points_default[(i + 1) % 20] - initial) / np.linalg.norm(
            points_default[(i + 1) % 20] - initial, ord=2
        )
        a_i = 2 * np.arcsin(np.linalg.norm(v_norm_ii - v_norm_i, ord=2) / 2)
        a_ii = 2 * np.arcsin(np.linalg.norm(v_norm_iii - v_norm_ii, ord=2) / 2)
        weight = (np.tan(a_i / 2) + np.tan(a_ii / 2)) / np.linalg.norm(
            points_default[i] - initial, ord=2
        )
        weights.append(weight)
        total_sum += weight

    return np.array(weights / total_sum)


def make_first_point_edge_plane(nodes, model_points):
    """Create first level points, edges, and planes."""
    points = [None for _ in range(72)]
    edges = [None for _ in range(140)]
    planes = [None for _ in range(70)]

    nowpoints = np.zeros(0)
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                if len(nowpoints) == 0:
                    nowpoints = np.array([[0.25 * i, 0.25 * j, 0.10 * k]])
                else:
                    nowpoints = np.append(
                        nowpoints, np.array([[0.25 * i, 0.25 * j, 0.10 * k]]), axis=0
                    )
    points[41] = initial_point(nowpoints, model_points, 0, 1, 2, 4, [-1, -1, -1])
    points[5] = initial_point(nowpoints, model_points, 1, 0, 3, 5, [-1, -1, 1])
    points[36] = initial_point(nowpoints, model_points, 2, 3, 0, 6, [-1, 1, -1])
    points[0] = initial_point(nowpoints, model_points, 3, 2, 1, 7, [-1, 1, 1])
    points[46] = initial_point(nowpoints, model_points, 4, 5, 0, 6, [1, -1, -1])
    points[10] = initial_point(nowpoints, model_points, 5, 4, 1, 7, [1, -1, 1])
    points[51] = initial_point(nowpoints, model_points, 6, 7, 2, 4, [1, 1, -1])
    points[15] = initial_point(nowpoints, model_points, 7, 6, 3, 5, [1, 1, 1])

    points[1] = initial_edge_point(
        nodes,
        np.array([-0.25, points[0].xyz[1] * 4 / 5 + points[5].xyz[1] * 1 / 5, 0.10]),
        np.array([0, 0, -1]),
        np.array([10, 0, 0]),
        [-1, 0, 1],
    )
    points[2] = initial_edge_point(
        nodes,
        np.array([-0.25, points[0].xyz[1] * 3 / 5 + points[5].xyz[1] * 2 / 5, 0.10]),
        np.array([0, 0, -1]),
        np.array([10, 0, 0]),
        [-1, 0, 1],
    )
    points[3] = initial_edge_point(
        nodes,
        np.array([-0.25, points[0].xyz[1] * 2 / 5 + points[5].xyz[1] * 3 / 5, 0.10]),
        np.array([0, 0, -1]),
        np.array([10, 0, 0]),
        [-1, 0, 1],
    )
    points[4] = initial_edge_point(
        nodes,
        np.array([-0.25, points[0].xyz[1] * 1 / 5 + points[5].xyz[1] * 4 / 5, 0.10]),
        np.array([0, 0, -1]),
        np.array([10, 0, 0]),
        [-1, 0, 1],
    )
    points[6] = initial_edge_point(
        nodes,
        np.array([points[5].xyz[0] * 4 / 5 + points[10].xyz[0] * 1 / 5, -0.25, 0.10]),
        np.array([0, 0, -1]),
        np.array([0, 10, 0]),
        [0, -1, 1],
    )
    points[7] = initial_edge_point(
        nodes,
        np.array([points[5].xyz[0] * 3 / 5 + points[10].xyz[0] * 2 / 5, -0.25, 0.10]),
        np.array([0, 0, -1]),
        np.array([0, 10, 0]),
        [0, -1, 1],
    )
    points[8] = initial_edge_point(
        nodes,
        np.array([points[5].xyz[0] * 2 / 5 + points[10].xyz[0] * 3 / 5, -0.25, 0.10]),
        np.array([0, 0, -1]),
        np.array([0, 10, 0]),
        [0, -1, 1],
    )
    points[9] = initial_edge_point(
        nodes,
        np.array([points[5].xyz[0] * 1 / 5 + points[10].xyz[0] * 4 / 5, -0.25, 0.10]),
        np.array([0, 0, -1]),
        np.array([0, 10, 0]),
        [0, -1, 1],
    )
    points[11] = initial_edge_point(
        nodes,
        np.array([0.25, points[10].xyz[1] * 4 / 5 + points[15].xyz[1] * 1 / 5, 0.10]),
        np.array([-10, 0, 0]),
        np.array([0, 0, -1]),
        [1, 0, 1],
    )
    points[12] = initial_edge_point(
        nodes,
        np.array([0.25, points[10].xyz[1] * 3 / 5 + points[15].xyz[1] * 2 / 5, 0.10]),
        np.array([-10, 0, 0]),
        np.array([0, 0, -1]),
        [1, 0, 1],
    )
    points[13] = initial_edge_point(
        nodes,
        np.array([0.25, points[10].xyz[1] * 2 / 5 + points[15].xyz[1] * 3 / 5, 0.10]),
        np.array([-10, 0, 0]),
        np.array([0, 0, -1]),
        [1, 0, 1],
    )
    points[14] = initial_edge_point(
        nodes,
        np.array([0.25, points[10].xyz[1] * 1 / 5 + points[15].xyz[1] * 4 / 5, 0.10]),
        np.array([-10, 0, 0]),
        np.array([0, 0, -1]),
        [1, 0, 1],
    )
    points[16] = initial_edge_point(
        nodes,
        np.array([points[15].xyz[0] * 4 / 5 + points[0].xyz[0] * 1 / 5, 0.25, 0.10]),
        np.array([0, -10, 0]),
        np.array([0, 0, -1]),
        [0, 1, 1],
    )
    points[17] = initial_edge_point(
        nodes,
        np.array([points[15].xyz[0] * 3 / 5 + points[0].xyz[0] * 2 / 5, 0.25, 0.10]),
        np.array([0, -10, 0]),
        np.array([0, 0, -1]),
        [0, 1, 1],
    )
    points[18] = initial_edge_point(
        nodes,
        np.array([points[15].xyz[0] * 2 / 5 + points[0].xyz[0] * 3 / 5, 0.25, 0.10]),
        np.array([0, -10, 0]),
        np.array([0, 0, -1]),
        [0, 1, 1],
    )
    points[19] = initial_edge_point(
        nodes,
        np.array([points[15].xyz[0] * 1 / 5 + points[0].xyz[0] * 4 / 5, 0.25, 0.10]),
        np.array([0, -10, 0]),
        np.array([0, 0, -1]),
        [0, 1, 1],
    )
    points[37] = initial_edge_point(
        nodes,
        np.array([-0.25, points[36].xyz[1] * 4 / 5 + points[41].xyz[1] * 1 / 5, -0.10]),
        np.array([10, 0, 0]),
        np.array([0, 0, 1]),
        [-1, 0, -1],
    )
    points[38] = initial_edge_point(
        nodes,
        np.array([-0.25, points[36].xyz[1] * 3 / 5 + points[41].xyz[1] * 2 / 5, -0.10]),
        np.array([10, 0, 0]),
        np.array([0, 0, 1]),
        [-1, 0, -1],
    )
    points[39] = initial_edge_point(
        nodes,
        np.array([-0.25, points[36].xyz[1] * 2 / 5 + points[41].xyz[1] * 3 / 5, -0.10]),
        np.array([10, 0, 0]),
        np.array([0, 0, 1]),
        [-1, 0, -1],
    )
    points[40] = initial_edge_point(
        nodes,
        np.array([-0.25, points[36].xyz[1] * 1 / 5 + points[41].xyz[1] * 4 / 5, -0.10]),
        np.array([10, 0, 0]),
        np.array([0, 0, 1]),
        [-1, 0, -1],
    )
    points[42] = initial_edge_point(
        nodes,
        np.array([points[41].xyz[0] * 4 / 5 + points[46].xyz[0] * 1 / 5, -0.25, -0.10]),
        np.array([0, 10, 0]),
        np.array([0, 0, 1]),
        [0, -1, -1],
    )
    points[43] = initial_edge_point(
        nodes,
        np.array([points[41].xyz[0] * 3 / 5 + points[46].xyz[0] * 2 / 5, -0.25, -0.10]),
        np.array([0, 10, 0]),
        np.array([0, 0, 1]),
        [0, -1, -1],
    )
    points[44] = initial_edge_point(
        nodes,
        np.array([points[41].xyz[0] * 2 / 5 + points[46].xyz[0] * 3 / 5, -0.25, -0.10]),
        np.array([0, 10, 0]),
        np.array([0, 0, 1]),
        [0, -1, -1],
    )
    points[45] = initial_edge_point(
        nodes,
        np.array([points[41].xyz[0] * 1 / 5 + points[46].xyz[0] * 4 / 5, -0.25, -0.10]),
        np.array([0, 10, 0]),
        np.array([0, 0, 1]),
        [0, -1, -1],
    )
    points[47] = initial_edge_point(
        nodes,
        np.array([0.25, points[46].xyz[1] * 4 / 5 + points[51].xyz[1] * 1 / 5, -0.10]),
        np.array([-10, 0, 0]),
        np.array([0, 0, 1]),
        [1, 0, -1],
    )
    points[48] = initial_edge_point(
        nodes,
        np.array([0.25, points[46].xyz[1] * 3 / 5 + points[51].xyz[1] * 2 / 5, -0.10]),
        np.array([-10, 0, 0]),
        np.array([0, 0, 1]),
        [1, 0, -1],
    )
    points[49] = initial_edge_point(
        nodes,
        np.array([0.25, points[46].xyz[1] * 2 / 5 + points[51].xyz[1] * 3 / 5, -0.10]),
        np.array([-10, 0, 0]),
        np.array([0, 0, 1]),
        [1, 0, -1],
    )
    points[50] = initial_edge_point(
        nodes,
        np.array([0.25, points[46].xyz[1] * 1 / 5 + points[51].xyz[1] * 4 / 5, -0.10]),
        np.array([-10, 0, 0]),
        np.array([0, 0, 1]),
        [1, 0, -1],
    )
    points[52] = initial_edge_point(
        nodes,
        np.array([points[51].xyz[0] * 4 / 5 + points[36].xyz[0] * 1 / 5, 0.25, -0.10]),
        np.array([0, -10, 0]),
        np.array([0, 0, 1]),
        [0, 1, -1],
    )
    points[53] = initial_edge_point(
        nodes,
        np.array([points[51].xyz[0] * 3 / 5 + points[36].xyz[0] * 2 / 5, 0.25, -0.10]),
        np.array([0, -10, 0]),
        np.array([0, 0, 1]),
        [0, 1, -1],
    )
    points[54] = initial_edge_point(
        nodes,
        np.array([points[51].xyz[0] * 2 / 5 + points[36].xyz[0] * 3 / 5, 0.25, -0.10]),
        np.array([0, -10, 0]),
        np.array([0, 0, 1]),
        [0, 1, -1],
    )
    points[55] = initial_edge_point(
        nodes,
        np.array([points[51].xyz[0] * 1 / 5 + points[36].xyz[0] * 4 / 5, 0.25, -0.10]),
        np.array([0, -10, 0]),
        np.array([0, 0, 1]),
        [0, 1, -1],
    )

    points_default = np.array(
        [
            [-0.25, -0.25 + 0.50 * 5 / 5],
            [-0.25, -0.25 + 0.50 * 4 / 5],
            [-0.25, -0.25 + 0.50 * 3 / 5],
            [-0.25, -0.25 + 0.50 * 2 / 5],
            [-0.25, -0.25 + 0.50 * 1 / 5],
            [-0.25 + 0.50 * 0 / 5, -0.25],
            [-0.25 + 0.50 * 1 / 5, -0.25],
            [-0.25 + 0.50 * 2 / 5, -0.25],
            [-0.25 + 0.50 * 3 / 5, -0.25],
            [-0.25 + 0.50 * 4 / 5, -0.25],
            [0.25, -0.25 + 0.50 * 0 / 5],
            [0.25, -0.25 + 0.50 * 1 / 5],
            [0.25, -0.25 + 0.50 * 2 / 5],
            [0.25, -0.25 + 0.50 * 3 / 5],
            [0.25, -0.25 + 0.50 * 4 / 5],
            [-0.25 + 0.50 * 5 / 5, 0.25],
            [-0.25 + 0.50 * 4 / 5, 0.25],
            [-0.25 + 0.50 * 3 / 5, 0.25],
            [-0.25 + 0.50 * 2 / 5, 0.25],
            [-0.25 + 0.50 * 1 / 5, 0.25],
        ]
    )
    top_array_x = np.array([points[0].xyz[0] - points_default[0][0]])
    top_array_y = np.array([points[0].xyz[1] - points_default[0][1]])
    for i in range(1, 20):
        top_array_x = np.append(top_array_x, [points[i].xyz[0] - points_default[i][0]])
        top_array_y = np.append(top_array_y, [points[i].xyz[1] - points_default[i][1]])
    bottom_array_x = np.array([points[36].xyz[0] - points_default[0][0]])
    bottom_array_y = np.array([points[36].xyz[1] - points_default[0][1]])
    for i in range(37, 56):
        bottom_array_x = np.append(
            bottom_array_x, [points[i].xyz[0] - points_default[i - 36][0]]
        )
        bottom_array_y = np.append(
            bottom_array_y, [points[i].xyz[1] - points_default[i - 36][1]]
        )

    points[56] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[57] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[58] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[59] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[60] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[61] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[62] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[63] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[64] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[65] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[66] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[67] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[68] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[69] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[70] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )
    points[71] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    bottom_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                np.dot(
                    bottom_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                -0.10,
            ]
        ),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    )

    points[20] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[21] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[22] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[23] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 1 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[24] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[25] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[26] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 1 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 1 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[27] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 2 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[28] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 3 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[29] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 4 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[30] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 3 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 3 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[31] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 2 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.25 + 0.50 * 2 / 5, -0.25 + 0.50 * 4 / 5])
                    ),
                )
                - 0.25
                + 0.50 * 4 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[32] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.15 + 0.30 * 2 / 5, -0.15 + 0.30 * 3 / 5])
                    ),
                )
                - 0.15
                + 0.30 * 2 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.15 + 0.30 * 2 / 5, -0.15 + 0.30 * 3 / 5])
                    ),
                )
                - 0.15
                + 0.30 * 3 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[33] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.15 + 0.30 * 2 / 5, -0.15 + 0.30 * 2 / 5])
                    ),
                )
                - 0.15
                + 0.30 * 2 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.15 + 0.30 * 2 / 5, -0.15 + 0.30 * 2 / 5])
                    ),
                )
                - 0.15
                + 0.30 * 2 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[34] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.15 + 0.30 * 3 / 5, -0.15 + 0.30 * 2 / 5])
                    ),
                )
                - 0.15
                + 0.30 * 3 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.15 + 0.30 * 3 / 5, -0.15 + 0.30 * 2 / 5])
                    ),
                )
                - 0.15
                + 0.30 * 2 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )
    points[35] = initial_plane_point(
        nodes,
        np.array(
            [
                np.dot(
                    top_array_x,
                    calculate_weight(
                        np.array([-0.15 + 0.30 * 3 / 5, -0.15 + 0.30 * 3 / 5])
                    ),
                )
                - 0.15
                + 0.30 * 3 / 5,
                np.dot(
                    top_array_y,
                    calculate_weight(
                        np.array([-0.15 + 0.30 * 3 / 5, -0.15 + 0.30 * 3 / 5])
                    ),
                )
                - 0.15
                + 0.30 * 3 / 5,
                0.10,
            ]
        ),
        np.array([0, 0, -1]),
        np.array([0, 0, 1]),
    )

    points_neighbor = default_config.points_neighbour

    for i in range(72):
        points[i].normal = define_initial_normal(points, i, points_neighbor[i])

    edges[0] = Edge(points, 0, 1)
    edges[1] = Edge(points, 1, 2)
    edges[2] = Edge(points, 2, 3)
    edges[3] = Edge(points, 3, 4)
    edges[4] = Edge(points, 4, 5)
    edges[5] = Edge(points, 19, 20)
    edges[6] = Edge(points, 20, 21)
    edges[7] = Edge(points, 21, 22)
    edges[8] = Edge(points, 22, 23)
    edges[9] = Edge(points, 23, 6)
    edges[10] = Edge(points, 18, 31)
    edges[11] = Edge(points, 31, 32)
    edges[12] = Edge(points, 32, 33)
    edges[13] = Edge(points, 33, 24)
    edges[14] = Edge(points, 24, 7)
    edges[15] = Edge(points, 17, 30)
    edges[16] = Edge(points, 30, 35)
    edges[17] = Edge(points, 35, 34)
    edges[18] = Edge(points, 34, 25)
    edges[19] = Edge(points, 25, 8)
    edges[20] = Edge(points, 16, 29)
    edges[21] = Edge(points, 29, 28)
    edges[22] = Edge(points, 28, 27)
    edges[23] = Edge(points, 27, 26)
    edges[24] = Edge(points, 26, 9)
    edges[25] = Edge(points, 15, 14)
    edges[26] = Edge(points, 14, 13)
    edges[27] = Edge(points, 13, 12)
    edges[28] = Edge(points, 12, 11)
    edges[29] = Edge(points, 11, 10)
    edges[30] = Edge(points, 0, 19)
    edges[31] = Edge(points, 19, 18)
    edges[32] = Edge(points, 18, 17)
    edges[33] = Edge(points, 17, 16)
    edges[34] = Edge(points, 16, 15)
    edges[35] = Edge(points, 1, 20)
    edges[36] = Edge(points, 20, 31)
    edges[37] = Edge(points, 31, 30)
    edges[38] = Edge(points, 30, 29)
    edges[39] = Edge(points, 29, 14)
    edges[40] = Edge(points, 2, 21)
    edges[41] = Edge(points, 21, 32)
    edges[42] = Edge(points, 32, 35)
    edges[43] = Edge(points, 35, 28)
    edges[44] = Edge(points, 28, 13)
    edges[45] = Edge(points, 3, 22)
    edges[46] = Edge(points, 22, 33)
    edges[47] = Edge(points, 33, 34)
    edges[48] = Edge(points, 34, 27)
    edges[49] = Edge(points, 27, 12)
    edges[50] = Edge(points, 4, 23)
    edges[51] = Edge(points, 23, 24)
    edges[52] = Edge(points, 24, 25)
    edges[53] = Edge(points, 25, 26)
    edges[54] = Edge(points, 26, 11)
    edges[55] = Edge(points, 5, 6)
    edges[56] = Edge(points, 6, 7)
    edges[57] = Edge(points, 7, 8)
    edges[58] = Edge(points, 8, 9)
    edges[59] = Edge(points, 9, 10)
    edges[60] = Edge(points, 0, 36)
    edges[61] = Edge(points, 1, 37)
    edges[62] = Edge(points, 2, 38)
    edges[63] = Edge(points, 3, 39)
    edges[64] = Edge(points, 4, 40)
    edges[65] = Edge(points, 5, 41)
    edges[66] = Edge(points, 6, 42)
    edges[67] = Edge(points, 7, 43)
    edges[68] = Edge(points, 8, 44)
    edges[69] = Edge(points, 9, 45)
    edges[70] = Edge(points, 10, 46)
    edges[71] = Edge(points, 11, 47)
    edges[72] = Edge(points, 12, 48)
    edges[73] = Edge(points, 13, 49)
    edges[74] = Edge(points, 14, 50)
    edges[75] = Edge(points, 15, 51)
    edges[76] = Edge(points, 16, 52)
    edges[77] = Edge(points, 17, 53)
    edges[78] = Edge(points, 18, 54)
    edges[79] = Edge(points, 19, 55)
    edges[80] = Edge(points, 36, 37)
    edges[81] = Edge(points, 37, 38)
    edges[82] = Edge(points, 38, 39)
    edges[83] = Edge(points, 39, 40)
    edges[84] = Edge(points, 40, 41)
    edges[85] = Edge(points, 55, 56)
    edges[86] = Edge(points, 56, 57)
    edges[87] = Edge(points, 57, 58)
    edges[88] = Edge(points, 58, 59)
    edges[89] = Edge(points, 59, 42)
    edges[90] = Edge(points, 54, 67)
    edges[91] = Edge(points, 67, 68)
    edges[92] = Edge(points, 68, 69)
    edges[93] = Edge(points, 69, 60)
    edges[94] = Edge(points, 60, 43)
    edges[95] = Edge(points, 53, 66)
    edges[96] = Edge(points, 66, 71)
    edges[97] = Edge(points, 71, 70)
    edges[98] = Edge(points, 70, 61)
    edges[99] = Edge(points, 61, 44)
    edges[100] = Edge(points, 52, 65)
    edges[101] = Edge(points, 65, 64)
    edges[102] = Edge(points, 64, 63)
    edges[103] = Edge(points, 63, 62)
    edges[104] = Edge(points, 62, 45)
    edges[105] = Edge(points, 51, 50)
    edges[106] = Edge(points, 50, 49)
    edges[107] = Edge(points, 49, 48)
    edges[108] = Edge(points, 48, 47)
    edges[109] = Edge(points, 47, 46)
    edges[110] = Edge(points, 36, 55)
    edges[111] = Edge(points, 55, 54)
    edges[112] = Edge(points, 54, 53)
    edges[113] = Edge(points, 53, 52)
    edges[114] = Edge(points, 52, 51)
    edges[115] = Edge(points, 37, 56)
    edges[116] = Edge(points, 56, 67)
    edges[117] = Edge(points, 67, 66)
    edges[118] = Edge(points, 66, 65)
    edges[119] = Edge(points, 65, 50)
    edges[120] = Edge(points, 38, 57)
    edges[121] = Edge(points, 57, 68)
    edges[122] = Edge(points, 68, 71)
    edges[123] = Edge(points, 71, 64)
    edges[124] = Edge(points, 64, 49)
    edges[125] = Edge(points, 39, 58)
    edges[126] = Edge(points, 58, 69)
    edges[127] = Edge(points, 69, 70)
    edges[128] = Edge(points, 70, 63)
    edges[129] = Edge(points, 63, 48)
    edges[130] = Edge(points, 40, 59)
    edges[131] = Edge(points, 59, 60)
    edges[132] = Edge(points, 60, 61)
    edges[133] = Edge(points, 61, 62)
    edges[134] = Edge(points, 62, 47)
    edges[135] = Edge(points, 41, 42)
    edges[136] = Edge(points, 42, 43)
    edges[137] = Edge(points, 43, 44)
    edges[138] = Edge(points, 44, 45)
    edges[139] = Edge(points, 45, 46)

    planes[0] = Plane(points, edges, [0, 1, 20, 19])
    planes[1] = Plane(points, edges, [1, 2, 21, 20])
    planes[2] = Plane(points, edges, [2, 3, 22, 21])
    planes[3] = Plane(points, edges, [3, 4, 23, 22])
    planes[4] = Plane(points, edges, [4, 5, 6, 23])
    planes[5] = Plane(points, edges, [19, 20, 31, 18])
    planes[6] = Plane(points, edges, [20, 21, 32, 31])
    planes[7] = Plane(points, edges, [21, 22, 33, 32])
    planes[8] = Plane(points, edges, [22, 23, 24, 33])
    planes[9] = Plane(points, edges, [23, 6, 7, 24])
    planes[10] = Plane(points, edges, [18, 31, 30, 17])
    planes[11] = Plane(points, edges, [31, 32, 35, 30])
    planes[12] = Plane(points, edges, [32, 33, 34, 35])
    planes[13] = Plane(points, edges, [33, 24, 25, 34])
    planes[14] = Plane(points, edges, [24, 7, 8, 25])
    planes[15] = Plane(points, edges, [17, 30, 29, 16])
    planes[16] = Plane(points, edges, [30, 35, 28, 29])
    planes[17] = Plane(points, edges, [35, 34, 27, 28])
    planes[18] = Plane(points, edges, [34, 25, 26, 27])
    planes[19] = Plane(points, edges, [25, 8, 9, 26])
    planes[20] = Plane(points, edges, [16, 29, 14, 15])
    planes[21] = Plane(points, edges, [29, 28, 13, 14])
    planes[22] = Plane(points, edges, [28, 27, 12, 13])
    planes[23] = Plane(points, edges, [27, 26, 11, 12])
    planes[24] = Plane(points, edges, [26, 9, 10, 11])
    planes[25] = Plane(points, edges, [0, 36, 37, 1])
    planes[26] = Plane(points, edges, [1, 37, 38, 2])
    planes[27] = Plane(points, edges, [2, 38, 39, 3])
    planes[28] = Plane(points, edges, [3, 39, 40, 4])
    planes[29] = Plane(points, edges, [4, 40, 41, 5])
    planes[30] = Plane(points, edges, [5, 41, 42, 6])
    planes[31] = Plane(points, edges, [6, 42, 43, 7])
    planes[32] = Plane(points, edges, [7, 43, 44, 8])
    planes[33] = Plane(points, edges, [8, 44, 45, 9])
    planes[34] = Plane(points, edges, [9, 45, 46, 10])
    planes[35] = Plane(points, edges, [10, 46, 47, 11])
    planes[36] = Plane(points, edges, [11, 47, 48, 12])
    planes[37] = Plane(points, edges, [12, 48, 49, 13])
    planes[38] = Plane(points, edges, [13, 49, 50, 14])
    planes[39] = Plane(points, edges, [14, 50, 51, 15])
    planes[40] = Plane(points, edges, [15, 51, 52, 16])
    planes[41] = Plane(points, edges, [16, 52, 53, 17])
    planes[42] = Plane(points, edges, [17, 53, 54, 18])
    planes[43] = Plane(points, edges, [18, 54, 55, 19])
    planes[44] = Plane(points, edges, [19, 55, 36, 0])
    planes[45] = Plane(points, edges, [36, 55, 56, 37])
    planes[46] = Plane(points, edges, [37, 56, 57, 38])
    planes[47] = Plane(points, edges, [38, 57, 58, 39])
    planes[48] = Plane(points, edges, [39, 58, 59, 40])
    planes[49] = Plane(points, edges, [40, 59, 42, 41])
    planes[50] = Plane(points, edges, [55, 54, 67, 56])
    planes[51] = Plane(points, edges, [56, 67, 68, 57])
    planes[52] = Plane(points, edges, [57, 68, 69, 58])
    planes[53] = Plane(points, edges, [58, 69, 60, 59])
    planes[54] = Plane(points, edges, [59, 60, 43, 42])
    planes[55] = Plane(points, edges, [54, 53, 66, 67])
    planes[56] = Plane(points, edges, [67, 66, 71, 68])
    planes[57] = Plane(points, edges, [68, 71, 70, 69])
    planes[58] = Plane(points, edges, [69, 70, 61, 60])
    planes[59] = Plane(points, edges, [60, 61, 44, 43])
    planes[60] = Plane(points, edges, [53, 52, 65, 66])
    planes[61] = Plane(points, edges, [66, 65, 64, 71])
    planes[62] = Plane(points, edges, [71, 64, 63, 70])
    planes[63] = Plane(points, edges, [70, 63, 62, 61])
    planes[64] = Plane(points, edges, [61, 62, 45, 44])
    planes[65] = Plane(points, edges, [52, 51, 50, 65])
    planes[66] = Plane(points, edges, [65, 50, 49, 64])
    planes[67] = Plane(points, edges, [64, 49, 48, 63])
    planes[68] = Plane(points, edges, [63, 48, 47, 62])
    planes[69] = Plane(points, edges, [62, 47, 46, 45])

    return points, edges, planes


def make_first_out_points(points, planes):
    """
    Determine the level-1 placement of predicted points (out_points).

    Parameters
    ----------
    points : list(class Point,)
        List of existing points
    planes : list(class Plane)
        List of existing planes

    Returns
    -------
    out_points : list(class Point_outside,)
        List of existing out_points.
    boxes : list(class Box,)
        List of existing boxes.
    edges_outside : list(class Edge_outside,)
        List of existing edges_outside.
    planes_outside : list(class Plane_outside,)
        List of existing planes_outside.

    """
    # 予測点とて配置する点の座標。元々はグリッド上に配置してある。
    x = config.x_init_prediction_point_list
    y = config.y_init_prediction_point_list
    z = config.z_init_prediction_point_list
    out_points = []

    # 点を配置していく時にその点がTXやRXと同じ位置にある時はmodeを変更し、対応する点のindexを埋め込んでいく。
    for i in range(10):
        for j in range(10):
            for k in range(9):
                out_point = Point_outside(np.array([x[i], y[j], z[k]]))
                if i == 2 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 41
                elif i == 2 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 5
                elif i == 2 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 5
                elif i == 2 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 41
                elif i == 2 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 40
                elif i == 2 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 4
                elif i == 2 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 4
                elif i == 2 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 40
                elif i == 2 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 39
                elif i == 2 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 3
                elif i == 2 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 3
                elif i == 2 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 39
                elif i == 2 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 38
                elif i == 2 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 2
                elif i == 2 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 2
                elif i == 2 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 38
                elif i == 2 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 37
                elif i == 2 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 1
                elif i == 2 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 1
                elif i == 2 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 37
                elif i == 2 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 36
                elif i == 2 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 0
                elif i == 2 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 0
                elif i == 2 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 36
                elif i == 3 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 42
                elif i == 3 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 6
                elif i == 3 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 6
                elif i == 3 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 42
                elif i == 3 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 59
                elif i == 3 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 23
                elif i == 3 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 23
                elif i == 3 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 59
                elif i == 3 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 58
                elif i == 3 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 22
                elif i == 3 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 22
                elif i == 3 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 58
                elif i == 3 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 57
                elif i == 3 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 21
                elif i == 3 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 21
                elif i == 3 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 57
                elif i == 3 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 56
                elif i == 3 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 20
                elif i == 3 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 20
                elif i == 3 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 56
                elif i == 3 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 55
                elif i == 3 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 19
                elif i == 3 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 19
                elif i == 3 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 55
                elif i == 4 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 43
                elif i == 4 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 7
                elif i == 4 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 7
                elif i == 4 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 43
                elif i == 4 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 60
                elif i == 4 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 24
                elif i == 4 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 24
                elif i == 4 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 60
                elif i == 4 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 69
                elif i == 4 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 33
                elif i == 4 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 33
                elif i == 4 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 69
                elif i == 4 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 68
                elif i == 4 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 32
                elif i == 4 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 32
                elif i == 4 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 68
                elif i == 4 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 67
                elif i == 4 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 31
                elif i == 4 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 31
                elif i == 4 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 67
                elif i == 4 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 54
                elif i == 4 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 18
                elif i == 4 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 18
                elif i == 4 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 54
                elif i == 5 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 44
                elif i == 5 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 8
                elif i == 5 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 8
                elif i == 5 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 44
                elif i == 5 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 61
                elif i == 5 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 25
                elif i == 5 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 25
                elif i == 5 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 61
                elif i == 5 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 70
                elif i == 5 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 34
                elif i == 5 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 34
                elif i == 5 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 70
                elif i == 5 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 71
                elif i == 5 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 35
                elif i == 5 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 35
                elif i == 5 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 71
                elif i == 5 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 66
                elif i == 5 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 30
                elif i == 5 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 30
                elif i == 5 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 66
                elif i == 5 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 53
                elif i == 5 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 17
                elif i == 5 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 17
                elif i == 5 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 53
                elif i == 6 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 45
                elif i == 6 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 9
                elif i == 6 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 9
                elif i == 6 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 45
                elif i == 6 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 62
                elif i == 6 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 26
                elif i == 6 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 26
                elif i == 6 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 62
                elif i == 6 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 63
                elif i == 6 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 27
                elif i == 6 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 27
                elif i == 6 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 63
                elif i == 6 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 64
                elif i == 6 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 28
                elif i == 6 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 28
                elif i == 6 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 64
                elif i == 6 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 65
                elif i == 6 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 29
                elif i == 6 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 29
                elif i == 6 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 65
                elif i == 6 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 52
                elif i == 6 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 16
                elif i == 6 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 16
                elif i == 6 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 52
                elif i == 7 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 46
                elif i == 7 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 10
                elif i == 7 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 10
                elif i == 7 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 46
                elif i == 7 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 47
                elif i == 7 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 11
                elif i == 7 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 11
                elif i == 7 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 47
                elif i == 7 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 48
                elif i == 7 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 12
                elif i == 7 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 12
                elif i == 7 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 48
                elif i == 7 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 49
                elif i == 7 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 13
                elif i == 7 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 13
                elif i == 7 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 49
                elif i == 7 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 50
                elif i == 7 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 14
                elif i == 7 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 14
                elif i == 7 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 50
                elif i == 7 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 51
                elif i == 7 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 15
                elif i == 7 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 15
                elif i == 7 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 51
                out_points.append(out_point)

    # フェライトシールドの点(next_point)の変化に合わせて点を移動させる
    next_point = np.array(
        [[points[i].xyz[0], points[i].xyz[1], points[i].xyz[2]] for i in range(72)]
    )
    for i in range(72):
        next_point = np.append(
            next_point,
            [[points[i].xyz[0], points[i].xyz[1], 0.05 - points[i].xyz[2]]],
            axis=0,
        )
    for i in range(900):
        if out_points[i].mode == 0:
            out_points[i].calculate_nextpoint(next_point)
        elif out_points[i].mode == 1:
            out_points[i].xyz = points[out_points[i].index].xyz
        elif out_points[i].mode == 2:
            out_points[i].xyz = np.array(
                [
                    points[out_points[i].index].xyz[0],
                    points[out_points[i].index].xyz[1],
                    0.05 - points[out_points[i].index].xyz[2],
                ]
            )

    # さらに分割するときに重要なBoxとそれに伴うEdge,Planeをここで作成
    boxes = []
    edges_outside = []
    planes_outside = []
    edge_list = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ]  # この順番はコード全体で前提としているので崩さない
    plane_list = [
        [0, 2, 3, 1],
        [0, 2, 6, 4],
        [0, 1, 5, 4],
        [1, 3, 7, 5],
        [2, 3, 7, 6],
        [4, 6, 7, 5],
    ]  # この順番はコード全体で前提としているので崩さない
    for i in range(9):
        for j in range(9):
            for k in range(8):
                # シールドの中にはboxは作らない。シールドの外の時のみboxをを作成
                if (k == 2 and i >= 2 and i <= 6 and j >= 2 and j <= 6) or (
                    k == 5 and i >= 2 and i <= 6 and j >= 2 and j <= 6
                ):
                    continue
                box = Box(
                    [
                        i * 90 + j * 9 + k,
                        i * 90 + j * 9 + (k + 1),
                        i * 90 + (j + 1) * 9 + k,
                        i * 90 + (j + 1) * 9 + k + 1,
                        (i + 1) * 90 + j * 9 + k,
                        (i + 1) * 90 + j * 9 + (k + 1),
                        (i + 1) * 90 + (j + 1) * 9 + k,
                        (i + 1) * 90 + (j + 1) * 9 + k + 1,
                    ]
                )

                index_for_plane = []
                for ii in range(8):
                    if (
                        out_points[box.node[ii]].mode == 1
                        or out_points[box.node[ii]].mode == 2
                    ):
                        index_for_plane.append(out_points[box.node[ii]].index)
                if len(index_for_plane) > 0 and len(index_for_plane) == 4:
                    flag = 0
                    for ii in range(len(planes)):
                        for jj in range(4):
                            for iii in range(4):
                                if planes[ii].node[iii] == index_for_plane[jj]:
                                    flag += 1
                                    break
                    if flag == 4:
                        box.plane = ii
                        break

                # boxのedgeを追加していく。ただし、すでに存在しているかどうかを毎回チェックしている
                for edge_l in edge_list:
                    flag = 0
                    for edge_index in range(len(edges_outside)):
                        if (
                            edges_outside[edge_index].node[0] == box.node[edge_l[0]]
                            and edges_outside[edge_index].node[1] == box.node[edge_l[1]]
                        ):
                            flag = 1
                            box.edge_node.append(edge_index)
                            break
                    if flag == 0:
                        edge_outside = Edge_outside(
                            box.node[edge_l[0]], box.node[edge_l[1]]
                        )
                        box.edge_node.append(len(edges_outside))
                        edges_outside.append(edge_outside)

                # boxの平面を追加していく。ただし、すでに存在しているかどうかを毎回チェックしている
                for plane_l in plane_list:
                    for plane_index in range(len(planes_outside)):
                        flag = 0
                        for jj in range(4):
                            for iii in range(4):
                                if (
                                    planes_outside[plane_index].node[jj]
                                    == box.node[plane_l[iii]]
                                ):
                                    flag += 1
                                    break
                        if flag == 4:
                            box.plane_node.append(plane_index)
                            break
                    if flag != 4:
                        plane_outside = Plane_outside(
                            box.node[plane_l[0]],
                            box.node[plane_l[1]],
                            box.node[plane_l[2]],
                            box.node[plane_l[3]],
                        )
                        for edge_index in range(len(edges_outside)):
                            if (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[0]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[1]]
                            ) or (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[1]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[0]]
                            ):
                                plane_outside.edge[0] = edge_index
                            elif (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[1]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[2]]
                            ) or (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[2]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[1]]
                            ):
                                plane_outside.edge[1] = edge_index
                            elif (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[2]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[3]]
                            ) or (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[3]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[2]]
                            ):
                                plane_outside.edge[2] = edge_index
                            elif (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[3]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[0]]
                            ) or (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[0]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[3]]
                            ):
                                plane_outside.edge[3] = edge_index

                        box.plane_node.append(len(planes_outside))
                        planes_outside.append(plane_outside)
                boxes.append(box)
    return out_points, boxes, edges_outside, planes_outside
