"""Functions for ray-object intersection and coordinate calculation."""

import numpy as np


class Ray:
    """Class representing a ray for intersection calculations."""

    def __init__(self):
        """Initialize ray with origin and direction."""
        self.org = np.zeros(3)
        self.dir = np.zeros(3)


class IntersectInformation:
    """Class for storing intersection information."""

    def __init__(self):
        """Initialize intersection information."""
        self.normal = np.zeros(3)
        self.distance = np.inf


def IntersectAABBvsRay(aabb, ray):
    """Check intersection between AABB and ray."""
    t_max = np.inf
    t_min = -np.inf

    for i in range(3):
        if abs(ray.dir[i]) > 1e-6:
            t1 = (aabb[0][i] - ray.org[i]) / ray.dir[i]
            t2 = (aabb[1][i] - ray.org[i]) / ray.dir[i]
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            t_max = min(t_max, t_far)
            t_min = max(t_min, t_near)
            if t_min > t_max or t_max < 0:
                return False
        else:
            if aabb[0][i] <= ray.org[i] and ray.org[i] <= aabb[1][i]:
                t1 = -np.inf
                t2 = np.inf
            else:
                return False
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            t_max = min(t_max, t_far)
            t_min = max(t_min, t_near)
            if t_min > t_max:
                return False
    return True


# triangleとrayの交差判定。それぞれBVHとこのファイルでclass定義
def ray_intersects_triangle(triangle, ray):
    """Check intersection between ray and triangle."""
    edge1 = triangle.triangle[1] - triangle.triangle[0]
    edge2 = triangle.triangle[2] - triangle.triangle[0]
    ray_cross_e2 = np.cross(ray.dir, edge2)
    det = edge1.dot(ray_cross_e2)

    if abs(det) < 1e-8:
        return False, None, None, None

    inv_det = 1.0 / det
    s = ray.org - triangle.triangle[0]
    u = inv_det * np.dot(s, ray_cross_e2)

    if u < -1e-12 or u > 1 + 1e-12:
        return False, None, None, None

    s_cross_e1 = np.cross(s, edge1)
    v = inv_det * np.dot(ray.dir, s_cross_e1)

    if v < -1e-12 or u + v > 1 + 1e-12:
        return False, None, None, None

    t = inv_det * np.dot(edge2, s_cross_e1)

    if t > 1e-8:
        out_intersection_point = ray.org + ray.dir * t
        distance = np.linalg.norm(ray.dir * t)
        normal = np.cross(edge2, edge1) / np.linalg.norm(np.cross(edge2, edge1), ord=2)
        return True, out_intersection_point, distance, normal
    return False, None, None, None


def Intersect(nodes, index, ray, info):
    """Find intersection with BVH nodes."""
    if IntersectAABBvsRay(nodes[index].bbox, ray):
        if nodes[index].children[0] != -1:
            childResult = None
            for i in range(2):
                result = Intersect(nodes, nodes[index].children[i], ray, info)
                if result is not None:
                    childResult = result
            if childResult is not None:
                return childResult
        else:
            result = None
            for t in nodes[index].polygons:
                ans, point, distance, normal = ray_intersects_triangle(t, ray)
                if (ans) and (distance < info.distance):
                    result = t
                    info.distance = distance
                    info.normal = normal
            if result is not None:
                return result
    return None


def Intersect_all(nodes, index, ray):
    """Find all intersections with BVH nodes."""
    result = []
    if IntersectAABBvsRay(nodes[index].bbox, ray):
        if nodes[index].children[0] != -1:
            childResult = []
            for i in range(2):
                result_now = Intersect_all(nodes, nodes[index].children[i], ray)
                if len(result_now) != 0:
                    for t in result_now:
                        childResult.append(t)
            if len(childResult) != 0:
                return childResult
        else:
            for t in nodes[index].polygons:
                ans, point, distance, normal = ray_intersects_triangle(t, ray)
                if ans:
                    result.append(t)
            if len(result) == 0:
                return result
    return result


def calculate_point_ray_hit(nodes, ray_org, ray_dir):
    """Calculate point where ray hits mesh."""
    hitInfo = IntersectInformation()
    ray = Ray()
    for i in range(3):
        if abs(ray_org[i]) < 1e-8:
            ray_org[i] = 0.0  # 浮動小数点のところでバグらないように処置
        ray.org[i] = ray_org[i]
        ray.dir[i] = ray_dir[i]
    hitTriangle = Intersect(nodes, 0, ray, hitInfo)
    ans = [False, None, None, None]
    if hitTriangle is not None:
        ans = ray_intersects_triangle(hitTriangle, ray)
    return ans


def calculate_point_ray_hit_all(nodes, ray_org, ray_dir):
    """Calculate all points where ray hits mesh."""
    ray = Ray()
    for i in range(3):
        ray.org[i] = ray_org[i]
        ray.dir[i] = ray_dir[i]
    hitTriangle = Intersect_all(nodes, 0, ray)
    ans = []
    if hitTriangle is not None:
        for t in hitTriangle:
            ans.append(ray_intersects_triangle(t, ray)[1])
    return ans


def judge_in_out_for_onevector(nodes, ray):
    """Judge if point is inside or outside mesh."""
    hitTriangle = calculate_point_ray_hit_all(nodes, ray.org, ray.dir)
    num = 0
    hitpoint = []
    for i in range(len(hitTriangle)):
        flag = 0
        for j in range(len(hitpoint)):
            if np.linalg.norm(hitpoint[j] - hitTriangle[i]) < 1e-6:
                flag = 1
        if flag == 0:
            hitpoint.append(hitTriangle[i])
            num += 1
    if num % 2 == 0:
        return 0
    return 1


def judge_in_out(nodes, point):
    """
    Judge whether the point is in, out, or on the ferrite shield.

    Parameters
    ----------
    nodes : list(class Point)
        The number to specify a ferrite shield.
    point : Point
        The point to judge in or out of 3-D shape.

    Returns
    -------
    int
        0: out of the ferrite shield
        1: in the ferrite shield
        2: on the surface of the ferrite shield

    """
    ray = Ray()
    for i in range(3):
        ray.org[i] = point[i]
    in_out = -1
    for i in range(3):
        for j in [-1, 1]:
            for k in range(3):
                if k == i:
                    ray.dir[k] = j
                else:
                    ray.dir[k] = 0
            if in_out == -1:
                in_out = judge_in_out_for_onevector(nodes, ray)
            elif in_out != judge_in_out_for_onevector(nodes, ray):
                in_out = 2

    return in_out
