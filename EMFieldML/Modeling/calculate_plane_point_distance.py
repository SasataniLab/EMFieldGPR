"""Calculate the distance between a plane and a point."""

import numpy as np


def distance_from_plane(point, a, b, c, d):
    """
    Calculate the distance between a plane and a point.

    Parameters
    ----------
    point : ndarray(3)
        Coordinates of the point.
    a, b, c, d : float
        Equation of the plane. ax + by + cz + d = 0

    Returns
    -------
    float
        The distance between a plane and a point

    """
    return abs(a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(
        a**2 + b**2 + c**2
    )


def make_plane_from_edge(point, pointA, pointB, pointC):
    """
    Calculate the equation of a plane that passes through "point" and is parallel to the plane that passes through "pointA", "pointB", and "pointC".

    Parameters
    ----------
    point : ndarray(3)
        Point through which the plane passes.
    pointA, pointB, pointC : ndarray(3)
        The plane will be parallel to the plane passing through these three points.

    Returns
    -------
    normal_vector[0], normal_vector[1], normal_vector[2], d : float
        They are a, b, c, d of the plane equation ax + by + cz + d = 0.

    """
    # ベクトルを計算
    vector1 = pointB - pointA
    vector2 = pointC - pointA

    # 2つのベクトルの外積を計算して法線ベクトルを得る
    normal_vector = np.cross(vector1, vector2)

    # 法線ベクトルを正規化
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # 平面の方程式 ax + by + cz + d = 0 の d を計算
    d = -point.dot(normal_vector)

    return normal_vector[0], normal_vector[1], normal_vector[2], d


def search_near_points_from_plane(points, a, b, c, d):
    """
    Find the point closest to the given plane from the set of points.

    Parameters
    ----------
    points : list(ndarray(3))
        the set of points.
    a, b, c, d : float
        Equation of the plane. ax + by + cz + d = 0

    Returns
    -------
    ans_distance : float
        Distance between the closest point and the plane.
    ans_point : ndarray(3)
        The closest point.

    """
    ans_distance = np.inf
    ans_point = None
    for point in points:
        distance = distance_from_plane(point, a, b, c, d)
        if distance < ans_distance:
            ans_distance = distance
            ans_point = point
    return ans_distance, ans_point
