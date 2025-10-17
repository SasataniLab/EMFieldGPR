"""Functions for creating Level-2 and Level-3 of Aligned Edge Polycube mesh and Adaptive Exterior Grid."""

import sys

import numpy as np

from EMFieldML.Modeling.RayIntersect import calculate_point_ray_hit, judge_in_out
from EMFieldML.Modeling.shared_geometry import (
    Box,
    Edge,
    Edge_outside,
    Plane,
    Plane_outside,
    Point,
    Point_outside,
)

"""Functions for creating Level-2 and Level-3 of Aligned Edge Polycube mesh and Adaptive Exterior Grid."""


def calculate_ray(vector, normal):
    """Calculate ray direction for edge intersection."""
    s = -(vector[0] * normal[0] + vector[1] * normal[1] + vector[2] * normal[2]) / (
        vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2
    )
    ray_dir = s * vector + normal
    return ray_dir / np.linalg.norm(ray_dir, ord=2)


def search_edge_point(nodes, points, edge):
    """Search for edge points with specific normal adjustments."""
    point1 = points[edge.node[0]].xyz
    point2 = points[edge.node[1]].xyz
    neutralpoint = (point1 + point2) / 2
    if edge.node[0] == 81 and edge.node[1] == 221:
        if edge.normal[0] < 0:
            edge.normal[0] *= -1
        if edge.normal[1] < 0:
            edge.normal[1] *= -1
    if edge.node[0] == 122 and edge.node[1] == 215:
        if edge.normal[0] < 0:
            edge.normal[0] *= -1
        if edge.normal[1] < 0:
            edge.normal[1] *= -1
    if edge.node[0] == 107 and edge.node[1] == 213:
        if edge.normal[0] < 0:
            edge.normal[0] *= -1
        if edge.normal[1] > 0:
            edge.normal[1] *= -1
    if edge.node[0] == 77 and edge.node[1] == 217:
        if edge.normal[0] < 0:
            edge.normal[0] *= -1
        if edge.normal[1] > 0:
            edge.normal[1] *= -1
    if edge.node[0] == 92 and edge.node[1] == 227:
        if edge.normal[0] > 0:
            edge.normal[0] *= -1
        if edge.normal[1] > 0:
            edge.normal[1] *= -1
    if edge.node[0] == 111 and edge.node[1] == 233:
        if edge.normal[0] > 0:
            edge.normal[0] *= -1
        if edge.normal[1] > 0:
            edge.normal[1] *= -1
    if edge.node[0] == 126 and edge.node[1] == 235:
        if edge.normal[0] > 0:
            edge.normal[0] *= -1
        if edge.normal[1] < 0:
            edge.normal[1] *= -1
    if edge.node[0] == 96 and edge.node[1] == 231:
        if edge.normal[0] > 0:
            edge.normal[0] *= -1
        if edge.normal[1] < 0:
            edge.normal[1] *= -1
    if neutralpoint[0] * edge.normal[0] + neutralpoint[1] * edge.normal[1] > 0:
        if (
            calculate_point_ray_hit(
                nodes, neutralpoint + edge.normal * 5e-4, edge.normal
            )[2]
            is not None
        ) and calculate_point_ray_hit(
            nodes, neutralpoint + edge.normal * 1e-4, edge.normal
        )[
            2
        ] < 1e-2:
            first_ray = edge.normal
            point1 = calculate_point_ray_hit(
                nodes, neutralpoint - first_ray * 1e-4, first_ray
            )[1]
            point2 = calculate_point_ray_hit(nodes, point1, first_ray)[1]
            if point2 is None:
                # print(neutralpoint + edge.normal * 1e-4,neutralpoint, first_ray, edge.node[0], edge.node[1], points[edge.node[0]].xyz, point1, point1 + first_ray * 1e-5)
                # sys.exit("point1")
                return point1
            ray_dir = np.cross(edge.normal, point2 - point1) / np.linalg.norm(
                np.cross(edge.normal, point2 - point1)
            )
            if ray_dir[2] * first_ray[2] > 0:
                ray_dir *= -1
        else:
            first_ray = np.cross(edge.normal, point2 - point1) / np.linalg.norm(
                np.cross(edge.normal, point2 - point1)
            )
            if first_ray[2] * edge.normal[2] > 0:
                first_ray *= -1
            point1 = calculate_point_ray_hit(
                nodes,
                neutralpoint
                - first_ray * 1e-4
                - np.array([neutralpoint[0], neutralpoint[1], 0.0])
                * 1e-4
                / np.linalg.norm(np.array([neutralpoint[0], neutralpoint[1], 0.0])),
                first_ray,
            )[1]
            point2 = calculate_point_ray_hit(
                nodes, point1 + first_ray * 1e-5, first_ray
            )[1]
            if point2 is None:
                return point1
                # print(judge_in_out(nodes, point1 + first_ray * 1e-5),neutralpoint + edge.normal * 5e-4,neutralpoint, edge.normal, edge.node[0], edge.node[1], points[edge.node[0]].xyz, points[edge.node[1]].xyz, points[edge.node[0]].normal, points[edge.node[1]].normal, edge.normal)
                # sys.exit("point2")
            ray_dir = edge.normal
    else:
        first_ray = np.cross(edge.normal, point2 - point1) / np.linalg.norm(
            np.cross(edge.normal, point2 - point1)
        )
        if first_ray[2] * edge.normal[2] < 0:
            first_ray *= -1
        point1 = calculate_point_ray_hit(
            nodes,
            neutralpoint
            - np.array([neutralpoint[0], neutralpoint[1], 0.0])
            * 1e-4
            / np.linalg.norm(np.array([neutralpoint[0], neutralpoint[1], 0.0])),
            -edge.normal + np.array([neutralpoint[0], neutralpoint[1], 0.0]),
        )[1]
        point2 = calculate_point_ray_hit(nodes, point1 + first_ray * 1e-5, first_ray)[1]
        if point2 is None:
            return point1
        ray_dir = edge.normal
    vector_ray = first_ray
    count = 0
    while count < 100:
        nextpoint = (point1 + point2) / 2
        hitpoint = calculate_point_ray_hit(nodes, nextpoint, ray_dir)[1]
        if hitpoint is None:
            print(
                edge.node[0],
                edge.node[1],
                point1,
                point2,
                calculate_point_ray_hit(
                    nodes,
                    neutralpoint,
                    -edge.normal + np.array([neutralpoint[0], neutralpoint[1], 0.0]),
                )[1],
            )
            sys.exit("error at make_points_from_parallel_edge. at while, in_out is 0")
        if (
            judge_in_out(nodes, hitpoint - vector_ray * 1e-4) == 1
            and judge_in_out(nodes, hitpoint + vector_ray * 1e-4) != 1
        ):
            point2 = hitpoint
            p1 = calculate_point_ray_hit(nodes, hitpoint, -vector_ray)[1]
            if p1 is not None:
                point1 = p1
            else:
                break
        elif (
            judge_in_out(nodes, hitpoint - vector_ray * 1e-4) != 1
            and judge_in_out(nodes, hitpoint + vector_ray * 1e-4) == 1
        ):
            point1 = hitpoint
            p2 = calculate_point_ray_hit(nodes, hitpoint, vector_ray)[1]
            if p2 is not None:
                point2 = p2
            else:
                break
        else:
            break
        if np.linalg.norm(point2 - point1) < 1e-4:
            break
        count += 1
    return hitpoint


def make_points_from_edge(nodes, points, edge):
    """Create points from edge with proper normal direction."""
    point1 = points[edge.node[0]].xyz
    point2 = points[edge.node[1]].xyz
    first_point = (point1 + point2) / 2
    in_out = judge_in_out(nodes, first_point)
    if in_out == 0:
        normal = -edge.normal
    elif in_out == 1:
        normal = edge.normal
    else:
        return first_point
    hitpoint = calculate_point_ray_hit(nodes, first_point, normal)
    return hitpoint[1]


def make_points_from_parallel_edge(nodes, points, edge):
    """Create points from parallel edge with vector calculations."""
    point1 = points[edge.node[0]].xyz
    point2 = points[edge.node[1]].xyz
    vector_edge = (point2 - point1) / np.linalg.norm(point2 - point1, ord=2)
    neutralpoint = (point1 + point2) / 2
    if (
        judge_in_out(nodes, point2 - vector_edge * 1e-3) == 0
        and judge_in_out(nodes, point1 + vector_edge * 1e-3) == 1
    ):
        neutralpoint = point1 + vector_edge * 1e-3
    elif (
        judge_in_out(nodes, point2 - vector_edge * 1e-3) == 1
        and judge_in_out(nodes, point1 + vector_edge * 1e-3) == 0
    ):
        neutralpoint = point2 - vector_edge * 1e-3
    else:
        neutralpoint = (point1 + point2) / 2
    ray_dir = calculate_ray(vector_edge, edge.normal)
    in_out = judge_in_out(nodes, neutralpoint)
    if in_out == 0:
        ray_dir *= -1
    elif in_out == 2:
        return neutralpoint
    hitpoint = calculate_point_ray_hit(nodes, neutralpoint, ray_dir)
    if not hitpoint[0]:
        sys.exit("There are any mistake in make_points_from_parallel_edge. in edge")
    if abs(np.dot(hitpoint[3], vector_edge)) < 3e-2:
        return hitpoint[1]
    count = 0
    nextpoint = neutralpoint
    while count < 100:
        if edge.node[0] == 17 and edge.node[1] == 30:
            nextpoint[0] -= 1e-6
        if edge.node[0] == 25 and edge.node[1] == 8:
            nextpoint[0] -= 1e-6
        if edge.node[0] == 2 and edge.node[1] == 21:
            nextpoint[1] -= 1e-6
        if edge.node[0] == 28 and edge.node[1] == 13:
            nextpoint[1] -= 1e-6
        hitpoint = calculate_point_ray_hit(nodes, nextpoint, ray_dir)
        if hitpoint[1] is None:
            print(edge.node[0], edge.node[1])
            sys.exit("error at make_points_from_parallel_edge. 0")
        if abs(np.dot(hitpoint[3], vector_edge)) < 3e-2:
            break
        if (
            judge_in_out(nodes, hitpoint[1] - vector_edge * 1e-4) == in_out
            and judge_in_out(nodes, hitpoint[1] + vector_edge * 1e-4) != in_out
        ):
            point2 = hitpoint[1]
            p1 = calculate_point_ray_hit(nodes, hitpoint[1], -vector_edge)[1]
            if p1 is not None:
                point1 = p1
            else:
                break
        elif (
            judge_in_out(nodes, hitpoint[1] - vector_edge * 1e-4) != in_out
            and judge_in_out(nodes, hitpoint[1] + vector_edge * 1e-4) == in_out
        ):
            point1 = hitpoint[1]
            p2 = calculate_point_ray_hit(nodes, hitpoint[1], vector_edge)[1]
            if p2 is not None:
                point2 = p2
            else:
                break
        else:
            break
        if np.linalg.norm(point2 - point1) < 1e-4:
            break
        count += 1
        nextpoint = (point1 + point2) / 2
    return hitpoint[1]


def check_totu(point1, point2, point3, point4):
    """Check if four points form a valid quadrilateral."""
    v1 = point2 - point1
    v2 = point3 - point1
    v3 = point4 - point1
    v4 = np.cross(v1, v2)
    a = np.array([[v1[0], v2[0], v4[0]], [v1[1], v2[1], v4[1]], [v1[2], v2[2], v4[2]]])
    b = np.linalg.inv(a)
    c = np.array([[v3[0]], [v3[1]], [v3[2]]])
    ans = np.dot(b, c)
    return not (ans[0] < 0 or ans[1] < 0 or ans[0] + ans[1] < 1)


def make_points_from_plane(nodes, points, edges, plane):
    """Create points from plane with validation."""
    if not check_totu(
        points[plane.node[0]].xyz,
        points[plane.node[1]].xyz,
        points[plane.node[3]].xyz,
        points[plane.node[2]].xyz,
    ):
        if np.linalg.norm(
            points[plane.node[0]].xyz - points[plane.node[2]].xyz, ord=2
        ) < np.linalg.norm(
            points[plane.node[1]].xyz - points[plane.node[3]].xyz, ord=2
        ):
            neutralpoint = (points[plane.node[0]].xyz + points[plane.node[2]].xyz) / 2
        else:
            neutralpoint = (points[plane.node[1]].xyz + points[plane.node[3]].xyz) / 2
    else:
        neutralpoint = (
            points[plane.node[0]].xyz
            + points[plane.node[1]].xyz
            + points[plane.node[2]].xyz
            + points[plane.node[3]].xyz
        ) / 4
    normal = (
        edges[plane.edge[0]].normal
        + edges[plane.edge[1]].normal
        + edges[plane.edge[2]].normal
        + edges[plane.edge[3]].normal
    )
    normal /= np.linalg.norm(normal)

    vector1 = (points[plane.node[0]].xyz - points[plane.node[2]].xyz) / np.linalg.norm(
        points[plane.node[0]].xyz - points[plane.node[2]].xyz, ord=2
    )
    vector2 = (points[plane.node[1]].xyz - points[plane.node[3]].xyz) / np.linalg.norm(
        points[plane.node[1]].xyz - points[plane.node[3]].xyz, ord=2
    )

    edge_point = [-1, -1, -1, -1]
    for i in range(4):
        if (
            edges[edges[plane.edge[i]].children[0]].node[0] == plane.node[i % 4]
            or edges[edges[plane.edge[i]].children[0]].node[0]
            == plane.node[(i + 1) % 4]
        ):
            edge_point[i] = edges[edges[plane.edge[i]].children[0]].node[1]
        else:
            edge_point[i] = edges[edges[plane.edge[i]].children[0]].node[0]
    vector3 = (points[edge_point[0]].xyz - points[edge_point[2]].xyz) / np.linalg.norm(
        points[edge_point[0]].xyz - points[edge_point[2]].xyz, ord=2
    )
    vector4 = (points[edge_point[1]].xyz - points[edge_point[3]].xyz) / np.linalg.norm(
        points[edge_point[1]].xyz - points[edge_point[3]].xyz, ord=2
    )

    #  平面とそれぞれの点の距離を計算し、8点が同一平面上にあるかどうかをチェックする
    cross = np.cross(
        points[plane.node[1]].xyz - points[plane.node[0]].xyz,
        points[plane.node[3]].xyz - points[plane.node[0]].xyz,
    )
    cross /= np.linalg.norm(cross)
    d = -np.dot(cross, points[plane.node[0]].xyz)
    r1 = abs(np.dot(cross, points[plane.node[2]].xyz) + d)
    r2 = abs(np.dot(cross, points[edge_point[0]].xyz) + d)
    r3 = abs(np.dot(cross, points[edge_point[1]].xyz) + d)
    r4 = abs(np.dot(cross, points[edge_point[2]].xyz) + d)
    r5 = abs(np.dot(cross, points[edge_point[3]].xyz) + d)
    r = max(r1, r2, r3, r4, r5)
    if r < 5e-4:
        max_vector = 5
    elif (
        abs(vector1[2]) > 5e-2
        and abs(vector2[2]) < 5e-2
        and abs(np.dot(vector1, np.array([0.0, 0.0, 1.0]))) < 0.999
    ):
        max_vector = 1
    elif (
        abs(vector2[2]) > 5e-2
        and abs(vector1[2]) < 5e-2
        and abs(np.dot(vector2, np.array([0.0, 0.0, 1.0]))) < 0.999
    ):
        max_vector = 2
    elif (
        abs(vector3[2]) > 5e-2
        and abs(vector4[2]) < 5e-2
        and abs(np.dot(vector3, np.array([0.0, 0.0, 1.0]))) < 0.999
    ):
        max_vector = 3
    elif (
        abs(vector4[2]) > 5e-2
        and abs(vector3[2]) < 5e-2
        and abs(np.dot(vector4, np.array([0.0, 0.0, 1.0]))) < 0.999
    ):
        max_vector = 4
    else:
        max_vector = 5

    if max_vector == 1:
        point1 = points[plane.node[0]].xyz
        point2 = points[plane.node[2]].xyz
        vector_edge = vector1
    elif max_vector == 2:
        point1 = points[plane.node[1]].xyz
        point2 = points[plane.node[3]].xyz
        vector_edge = vector2
    elif max_vector == 3:
        point1 = points[edge_point[0]].xyz
        point2 = points[edge_point[2]].xyz
        vector_edge = vector3
    elif max_vector == 4:
        point1 = points[edge_point[1]].xyz
        point2 = points[edge_point[3]].xyz
        vector_edge = vector4
    else:
        ray_dir = np.cross(vector1, vector2)
        ray_dir /= np.linalg.norm(ray_dir)
        in_out = judge_in_out(nodes, neutralpoint)
        if in_out == 2:
            return neutralpoint, normal, 0
        if in_out == 0:
            ray_dir *= -1
        hitpoint = calculate_point_ray_hit(nodes, neutralpoint, ray_dir)
        if not hitpoint[0]:
            sys.exit("error in makeing plane point process.")
        else:
            return hitpoint[1], normal, 0

    if (
        judge_in_out(nodes, point2 + vector_edge * 3e-3) == 0
        and judge_in_out(nodes, point1 - vector_edge * 3e-3) == 1
    ):
        neutralpoint = point1 - vector_edge * 3e-3
    elif (
        judge_in_out(nodes, point2 + vector_edge * 3e-3) == 1
        and judge_in_out(nodes, point1 - vector_edge * 3e-3) == 0
    ):
        neutralpoint = point2 + vector_edge * 3e-3
    else:
        neutralpoint = (point1 + point2) / 2
    ray_dir = calculate_ray(vector_edge, normal)
    in_out = judge_in_out(nodes, neutralpoint)
    if in_out == 0:
        ray_dir *= -1
    elif in_out == 2:
        return neutralpoint, ray_dir, 0
    hitpoint = calculate_point_ray_hit(nodes, neutralpoint, ray_dir)
    if not hitpoint[0]:
        sys.exit("There are any mistake in make_points_from_parallel_edge. in plane")
    if abs(np.dot(hitpoint[3], vector_edge)) < 3e-2:
        return hitpoint[1], ray_dir, 0
    count = 0
    nextpoint = neutralpoint
    while count < 100:
        hitpoint = calculate_point_ray_hit(nodes, nextpoint, ray_dir)
        if hitpoint[1] is None:
            print(plane.node[0], nextpoint, ray_dir, vector_edge)
            sys.exit("error at make_points_from_parallel_edge. 1")
        if abs(np.dot(hitpoint[3], vector_edge)) < 3e-2:
            break
        if (
            judge_in_out(nodes, hitpoint[1] - vector_edge * 1e-5) == in_out
            and judge_in_out(nodes, hitpoint[1] + vector_edge * 1e-5) != in_out
        ):
            point2 = hitpoint[1]
            p1 = calculate_point_ray_hit(nodes, hitpoint[1], -vector_edge)[1]
            if p1 is not None:
                point1 = p1
            else:
                break
        elif (
            judge_in_out(nodes, hitpoint[1] - vector_edge * 1e-5) != in_out
            and judge_in_out(nodes, hitpoint[1] + vector_edge * 1e-5) == in_out
        ):
            point1 = hitpoint[1]
            p2 = calculate_point_ray_hit(nodes, hitpoint[1], vector_edge)[1]
            if p2 is not None:
                point2 = p2
            else:
                break
        else:
            break
        if np.linalg.norm(point2 - point1) < 1e-4:
            break
        nextpoint = (point1 + point2) / 2
        count += 1
    return hitpoint[1], ray_dir, 1


def set_boxes(box, out_points, edges_outside, planes_outside):
    """Set box properties with edges and planes."""
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
    ]
    plane_list = [
        [0, 2, 3, 1],
        [0, 2, 6, 4],
        [0, 1, 5, 4],
        [1, 3, 7, 5],
        [2, 3, 7, 6],
        [4, 6, 7, 5],
    ]
    for edge_l in edge_list:
        now_list = [len(out_points[box.node[edge_l[i]]].edges) for i in range(2)]
        focus = now_list.index(min(now_list))
        flag = 0
        for edge_index in out_points[box.node[edge_l[focus]]].edges:
            if (
                edges_outside[edge_index].node[0] == box.node[edge_l[0]]
                and edges_outside[edge_index].node[1] == box.node[edge_l[1]]
            ) or (
                edges_outside[edge_index].node[0] == box.node[edge_l[1]]
                and edges_outside[edge_index].node[1] == box.node[edge_l[0]]
            ):
                flag = 1
                box.edge_node.append(edge_index)
                break
        if flag == 0:
            sys.exit("edge of box is none.")

    for plane_l in plane_list:
        now_list = [len(out_points[box.node[plane_l[i]]].planes) for i in range(4)]
        focus = now_list.index(min(now_list))
        for plane_index in out_points[box.node[plane_l[focus]]].planes:
            flag = 0
            for jj in range(4):
                for iii in range(4):
                    if planes_outside[plane_index].node[jj] == box.node[plane_l[iii]]:
                        flag += 1
                        break
            if flag == 4:
                box.plane_node.append(plane_index)
                break
        if flag != 4:
            sys.exit("plane of box is none.")


def make_second_grid(nodes, points, edges, planes):
    """Create second level grid from existing mesh elements."""
    for i in range(len(edges)):
        if edges[i].children[0] == -1:
            count = 0
            for j in range(3):
                if abs(edges[i].normal[j]) > 5e-2:
                    count += 1
            if (
                points[edges[i].node[0]].corner == 1
                and points[edges[i].node[1]].corner == 1
                and abs(
                    points[edges[i].node[0]].xyz[2] - points[edges[i].node[1]].xyz[2]
                )
                < 1e-4
                and abs(
                    np.sqrt(
                        points[edges[i].node[0]].xyz[0] ** 2
                        + points[edges[i].node[0]].xyz[1] ** 2
                    )
                    - np.sqrt(
                        points[edges[i].node[1]].xyz[0] ** 2
                        + points[edges[i].node[1]].xyz[1] ** 2
                    )
                )
                < 5e-3
            ):
                newpoint = Point(search_edge_point(nodes, points, edges[i]))
                newpoint.corner = 1
            elif (
                count >= 2
                and abs(
                    np.dot(
                        (points[edges[i].node[0]].xyz - points[edges[i].node[1]].xyz)
                        / np.linalg.norm(
                            points[edges[i].node[0]].xyz - points[edges[i].node[1]].xyz
                        ),
                        np.array([0.0, 0.0, 1]),
                    )
                )
                > 3e-2
                and abs(
                    np.dot(
                        (points[edges[i].node[0]].xyz - points[edges[i].node[1]].xyz)
                        / np.linalg.norm(
                            points[edges[i].node[0]].xyz - points[edges[i].node[1]].xyz
                        ),
                        np.array([0.0, 0.0, 1]),
                    )
                )
                < 999e-3
            ):
                newpoint = Point(
                    make_points_from_parallel_edge(nodes, points, edges[i])
                )
                newpoint.corner = 1
            else:
                newpoint = Point(make_points_from_edge(nodes, points, edges[i]))
            newpoint.normal = edges[i].normal
            points.append(newpoint)
            newedge1 = Edge(points, edges[i].node[0], len(points) - 1)
            newedge2 = Edge(points, edges[i].node[1], len(points) - 1)
            edges.append(newedge1)
            edges[i].children[0] = len(edges) - 1
            edges.append(newedge2)
            edges[i].children[1] = len(edges) - 1

    for i in range(len(planes)):
        if planes[i].children[0] == -1:
            newpoint, normal, corner = make_points_from_plane(
                nodes, points, edges, planes[i]
            )
            newpoint = Point(newpoint)
            newpoint.normal = normal
            newpoint.corner = corner
            points.append(newpoint)
            plane_edge_point = [-1, -1, -1, -1]
            if (
                edges[edges[planes[i].edge[0]].children[0]].node[0] == planes[i].node[0]
                or edges[edges[planes[i].edge[0]].children[0]].node[0]
                == planes[i].node[1]
            ):
                plane_edge_point[0] = edges[edges[planes[i].edge[0]].children[0]].node[
                    1
                ]
            else:
                plane_edge_point[0] = edges[edges[planes[i].edge[0]].children[0]].node[
                    0
                ]
            if (
                edges[edges[planes[i].edge[1]].children[0]].node[0] == planes[i].node[1]
                or edges[edges[planes[i].edge[1]].children[0]].node[0]
                == planes[i].node[2]
            ):
                plane_edge_point[1] = edges[edges[planes[i].edge[1]].children[0]].node[
                    1
                ]
            else:
                plane_edge_point[1] = edges[edges[planes[i].edge[1]].children[0]].node[
                    0
                ]
            if (
                edges[edges[planes[i].edge[2]].children[0]].node[0] == planes[i].node[2]
                or edges[edges[planes[i].edge[2]].children[0]].node[0]
                == planes[i].node[3]
            ):
                plane_edge_point[2] = edges[edges[planes[i].edge[2]].children[0]].node[
                    1
                ]
            else:
                plane_edge_point[2] = edges[edges[planes[i].edge[2]].children[0]].node[
                    0
                ]
            if (
                edges[edges[planes[i].edge[3]].children[0]].node[0] == planes[i].node[3]
                or edges[edges[planes[i].edge[3]].children[0]].node[0]
                == planes[i].node[0]
            ):
                plane_edge_point[3] = edges[edges[planes[i].edge[3]].children[0]].node[
                    1
                ]
            else:
                plane_edge_point[3] = edges[edges[planes[i].edge[3]].children[0]].node[
                    0
                ]

            edges.append(Edge(points, plane_edge_point[0], len(points) - 1))
            edges.append(Edge(points, plane_edge_point[1], len(points) - 1))
            edges.append(Edge(points, plane_edge_point[2], len(points) - 1))
            edges.append(Edge(points, plane_edge_point[3], len(points) - 1))

            planes[i].children = [
                len(planes),
                len(planes) + 1,
                len(planes) + 2,
                len(planes) + 3,
            ]
            planes.append(
                Plane(
                    points,
                    edges,
                    [
                        planes[i].node[0],
                        plane_edge_point[0],
                        len(points) - 1,
                        plane_edge_point[3],
                    ],
                )
            )
            planes.append(
                Plane(
                    points,
                    edges,
                    [
                        planes[i].node[1],
                        plane_edge_point[1],
                        len(points) - 1,
                        plane_edge_point[0],
                    ],
                )
            )
            planes.append(
                Plane(
                    points,
                    edges,
                    [
                        planes[i].node[2],
                        plane_edge_point[2],
                        len(points) - 1,
                        plane_edge_point[1],
                    ],
                )
            )
            planes.append(
                Plane(
                    points,
                    edges,
                    [
                        planes[i].node[3],
                        plane_edge_point[3],
                        len(points) - 1,
                        plane_edge_point[2],
                    ],
                )
            )
    return points, edges, planes


def make_second_out_points(
    out_points, boxes, edges_outside, planes_outside, points, edges, planes
):
    """Create second level outside points from existing elements."""
    for i in range(len(edges_outside)):
        if edges_outside[i].children[0] == -1:
            point_index1 = edges_outside[i].node[0]
            point_index2 = edges_outside[i].node[1]
            if (
                out_points[point_index1].mode == 1
                and out_points[point_index2].mode == 1
            ):
                new_point_index1 = out_points[point_index1].index
                new_point_index2 = out_points[point_index2].index
                for j in range(len(edges)):
                    if (
                        edges[j].node[0] == new_point_index1
                        and edges[j].node[1] == new_point_index2
                    ) or (
                        edges[j].node[0] == new_point_index2
                        and edges[j].node[1] == new_point_index1
                    ):
                        edge_index1 = edges[j].children[0]
                        true_point_index = max(
                            edges[edge_index1].node[0], edges[edge_index1].node[1]
                        )
                        point = Point_outside(points[true_point_index].xyz)
                        point.index = true_point_index
                        point.mode = out_points[point_index1].mode
                        break
            elif (
                out_points[point_index1].mode == 2
                and out_points[point_index2].mode == 2
            ):
                new_point_index1 = out_points[point_index1].index
                new_point_index2 = out_points[point_index2].index
                for j in range(len(edges)):
                    if (
                        edges[j].node[0] == new_point_index1
                        and edges[j].node[1] == new_point_index2
                    ) or (
                        edges[j].node[0] == new_point_index2
                        and edges[j].node[1] == new_point_index1
                    ):
                        edge_index1 = edges[j].children[0]
                        true_point_index = max(
                            edges[edge_index1].node[0], edges[edge_index1].node[1]
                        )
                        point = Point_outside(
                            [
                                points[true_point_index].xyz[0],
                                points[true_point_index].xyz[1],
                                0.05 - points[true_point_index].xyz[2],
                            ]
                        )
                        point.index = true_point_index
                        point.mode = out_points[point_index1].mode
                        break
            else:
                point = Point_outside(
                    (out_points[point_index1].xyz + out_points[point_index2].xyz) / 2
                )

            edges_outside[i].children = [len(edges_outside), len(edges_outside) + 1]
            edges_outside[i].center = len(out_points)
            edges_outside.append(Edge_outside(point_index1, len(out_points)))
            edges_outside.append(Edge_outside(point_index2, len(out_points)))
            point.parent.append(point_index1)
            point.parent.append(point_index2)
            out_points.append(point)

    for i in range(len(planes_outside)):
        if planes_outside[i].children[0] == -1:
            point_index1 = planes_outside[i].node[0]
            point_index2 = planes_outside[i].node[1]
            point_index3 = planes_outside[i].node[2]
            point_index4 = planes_outside[i].node[3]
            point_index5 = edges_outside[planes_outside[i].edge[0]].center
            point_index6 = edges_outside[planes_outside[i].edge[1]].center
            point_index7 = edges_outside[planes_outside[i].edge[2]].center
            point_index8 = edges_outside[planes_outside[i].edge[3]].center
            if (
                edges_outside[
                    edges_outside[planes_outside[i].edge[0]].children[0]
                ].node[0]
                == point_index1
                or edges_outside[
                    edges_outside[planes_outside[i].edge[0]].children[0]
                ].node[1]
                == point_index1
            ):
                edge_index1 = edges_outside[planes_outside[i].edge[0]].children[0]
                edge_index2 = edges_outside[planes_outside[i].edge[0]].children[1]
            else:
                edge_index2 = edges_outside[planes_outside[i].edge[0]].children[0]
                edge_index1 = edges_outside[planes_outside[i].edge[0]].children[1]
            if (
                edges_outside[
                    edges_outside[planes_outside[i].edge[1]].children[0]
                ].node[0]
                == point_index2
                or edges_outside[
                    edges_outside[planes_outside[i].edge[1]].children[0]
                ].node[1]
                == point_index2
            ):
                edge_index3 = edges_outside[planes_outside[i].edge[1]].children[0]
                edge_index4 = edges_outside[planes_outside[i].edge[1]].children[1]
            else:
                edge_index4 = edges_outside[planes_outside[i].edge[1]].children[0]
                edge_index3 = edges_outside[planes_outside[i].edge[1]].children[1]
            if (
                edges_outside[
                    edges_outside[planes_outside[i].edge[2]].children[0]
                ].node[0]
                == point_index3
                or edges_outside[
                    edges_outside[planes_outside[i].edge[2]].children[0]
                ].node[1]
                == point_index3
            ):
                edge_index5 = edges_outside[planes_outside[i].edge[2]].children[0]
                edge_index6 = edges_outside[planes_outside[i].edge[2]].children[1]
            else:
                edge_index6 = edges_outside[planes_outside[i].edge[2]].children[0]
                edge_index5 = edges_outside[planes_outside[i].edge[2]].children[1]
            if (
                edges_outside[
                    edges_outside[planes_outside[i].edge[3]].children[0]
                ].node[0]
                == point_index4
                or edges_outside[
                    edges_outside[planes_outside[i].edge[3]].children[0]
                ].node[1]
                == point_index4
            ):
                edge_index7 = edges_outside[planes_outside[i].edge[3]].children[0]
                edge_index8 = edges_outside[planes_outside[i].edge[3]].children[1]
            else:
                edge_index8 = edges_outside[planes_outside[i].edge[3]].children[0]
                edge_index7 = edges_outside[planes_outside[i].edge[3]].children[1]
            if (
                out_points[point_index1].mode == 1
                and out_points[point_index2].mode == 1
                and out_points[point_index3].mode == 1
                and out_points[point_index4].mode == 1
            ):
                new_point_index = [
                    out_points[point_index1].index,
                    out_points[point_index2].index,
                    out_points[point_index3].index,
                    out_points[point_index4].index,
                ]
                for j in range(len(planes)):
                    count = 0
                    for k in range(4):
                        for m in range(4):
                            if planes[j].node[k] == new_point_index[m]:
                                count += 1
                    if count == 4:
                        plane_index1 = planes[j].children[0]
                        true_point_index = max(
                            planes[plane_index1].node[0],
                            planes[plane_index1].node[1],
                            planes[plane_index1].node[2],
                            planes[plane_index1].node[3],
                        )
                        point = Point_outside(points[true_point_index].xyz)
                        point.index = true_point_index
                        point.mode = out_points[point_index1].mode
                        break
            elif (
                out_points[point_index1].mode == 2
                and out_points[point_index2].mode == 2
                and out_points[point_index3].mode == 2
                and out_points[point_index4].mode == 2
            ):
                new_point_index = [
                    out_points[point_index1].index,
                    out_points[point_index2].index,
                    out_points[point_index3].index,
                    out_points[point_index4].index,
                ]
                for j in range(len(planes)):
                    count = 0
                    for k in range(4):
                        for m in range(4):
                            if planes[j].node[k] == new_point_index[m]:
                                count += 1
                    if count == 4:
                        plane_index1 = planes[j].children[0]
                        true_point_index = max(
                            planes[plane_index1].node[0],
                            planes[plane_index1].node[1],
                            planes[plane_index1].node[2],
                            planes[plane_index1].node[3],
                        )
                        point = Point_outside(
                            [
                                points[true_point_index].xyz[0],
                                points[true_point_index].xyz[1],
                                0.05 - points[true_point_index].xyz[2],
                            ]
                        )
                        point.index = true_point_index
                        point.mode = out_points[point_index1].mode
                        break
            else:
                point = Point_outside(
                    (
                        out_points[point_index5].xyz
                        + out_points[point_index6].xyz
                        + out_points[point_index7].xyz
                        + out_points[point_index8].xyz
                    )
                    / 4
                )

            planes_outside[i].children = [len(planes_outside) + ii for ii in range(4)]
            planes_outside[i].center = len(out_points)
            plane1 = Plane_outside(
                point_index1, point_index5, len(out_points), point_index8
            )
            plane1.edge = [
                edge_index1,
                len(edges_outside),
                len(edges_outside) + 3,
                edge_index8,
            ]
            plane2 = Plane_outside(
                point_index5, point_index2, point_index6, len(out_points)
            )
            plane2.edge = [
                edge_index2,
                edge_index3,
                len(edges_outside) + 1,
                len(edges_outside),
            ]
            plane3 = Plane_outside(
                len(out_points), point_index6, point_index3, point_index7
            )
            plane3.edge = [
                len(edges_outside) + 1,
                edge_index4,
                edge_index5,
                len(edges_outside) + 2,
            ]
            plane4 = Plane_outside(
                point_index8, len(out_points), point_index7, point_index4
            )
            plane4.edge = [
                len(edges_outside) + 3,
                len(edges_outside) + 2,
                edge_index6,
                edge_index7,
            ]
            planes_outside.append(plane1)
            planes_outside.append(plane2)
            planes_outside.append(plane3)
            planes_outside.append(plane4)
            edges_outside.append(Edge_outside(point_index5, len(out_points)))
            edges_outside.append(Edge_outside(point_index6, len(out_points)))
            edges_outside.append(Edge_outside(point_index7, len(out_points)))
            edges_outside.append(Edge_outside(point_index8, len(out_points)))
            point.parent.append(point_index1)
            point.parent.append(point_index2)
            point.parent.append(point_index3)
            point.parent.append(point_index4)
            out_points.append(point)

    for i in range(len(edges_outside)):
        point1 = out_points[edges_outside[i].node[0]]
        point2 = out_points[edges_outside[i].node[1]]
        if i not in point1.edges:
            point1.edges.append(i)
        if i not in point2.edges:
            point2.edges.append(i)

    for i in range(len(planes_outside)):
        point1 = out_points[planes_outside[i].node[0]]
        point2 = out_points[planes_outside[i].node[1]]
        point3 = out_points[planes_outside[i].node[2]]
        point4 = out_points[planes_outside[i].node[3]]
        if i not in point1.planes:
            point1.planes.append(i)
        if i not in point2.planes:
            point2.planes.append(i)
        if i not in point3.planes:
            point3.planes.append(i)
        if i not in point4.planes:
            point4.planes.append(i)

    for i in range(len(boxes)):
        if boxes[i].children[0] == -1:
            position = np.array([0.0, 0.0, 0.0])
            for j in range(len(boxes[i].plane_node)):
                position += out_points[
                    planes_outside[boxes[i].plane_node[j]].center
                ].xyz
                edges_outside.append(
                    Edge_outside(
                        planes_outside[boxes[i].plane_node[j]].center, len(out_points)
                    )
                )
                out_points[planes_outside[boxes[i].plane_node[j]].center].edges.append(
                    len(edges_outside) - 1
                )
            position = position / 6.0
            point = Point_outside(position)
            for j in range(len(boxes[i].plane_node)):
                point.parent.append(planes_outside[boxes[i].plane_node[j]].center)
                point.edges.append(len(edges_outside) - len(boxes[i].plane_node) + j)
            out_points.append(point)
            box0 = Box(
                [
                    boxes[i].node[0],
                    edges_outside[boxes[i].edge_node[0]].center,
                    edges_outside[boxes[i].edge_node[1]].center,
                    planes_outside[boxes[i].plane_node[0]].center,
                    edges_outside[boxes[i].edge_node[2]].center,
                    planes_outside[boxes[i].plane_node[2]].center,
                    planes_outside[boxes[i].plane_node[1]].center,
                    len(out_points) - 1,
                ]
            )
            box1 = Box(
                [
                    edges_outside[boxes[i].edge_node[0]].center,
                    boxes[i].node[1],
                    planes_outside[boxes[i].plane_node[0]].center,
                    edges_outside[boxes[i].edge_node[3]].center,
                    planes_outside[boxes[i].plane_node[2]].center,
                    edges_outside[boxes[i].edge_node[4]].center,
                    len(out_points) - 1,
                    planes_outside[boxes[i].plane_node[3]].center,
                ]
            )
            box2 = Box(
                [
                    edges_outside[boxes[i].edge_node[1]].center,
                    planes_outside[boxes[i].plane_node[0]].center,
                    boxes[i].node[2],
                    edges_outside[boxes[i].edge_node[5]].center,
                    planes_outside[boxes[i].plane_node[1]].center,
                    len(out_points) - 1,
                    edges_outside[boxes[i].edge_node[6]].center,
                    planes_outside[boxes[i].plane_node[4]].center,
                ]
            )
            box3 = Box(
                [
                    planes_outside[boxes[i].plane_node[0]].center,
                    edges_outside[boxes[i].edge_node[3]].center,
                    edges_outside[boxes[i].edge_node[5]].center,
                    boxes[i].node[3],
                    len(out_points) - 1,
                    planes_outside[boxes[i].plane_node[3]].center,
                    planes_outside[boxes[i].plane_node[4]].center,
                    edges_outside[boxes[i].edge_node[7]].center,
                ]
            )
            box4 = Box(
                [
                    edges_outside[boxes[i].edge_node[2]].center,
                    planes_outside[boxes[i].plane_node[2]].center,
                    planes_outside[boxes[i].plane_node[1]].center,
                    len(out_points) - 1,
                    boxes[i].node[4],
                    edges_outside[boxes[i].edge_node[8]].center,
                    edges_outside[boxes[i].edge_node[9]].center,
                    planes_outside[boxes[i].plane_node[5]].center,
                ]
            )
            box5 = Box(
                [
                    planes_outside[boxes[i].plane_node[2]].center,
                    edges_outside[boxes[i].edge_node[4]].center,
                    len(out_points) - 1,
                    planes_outside[boxes[i].plane_node[3]].center,
                    edges_outside[boxes[i].edge_node[8]].center,
                    boxes[i].node[5],
                    planes_outside[boxes[i].plane_node[5]].center,
                    edges_outside[boxes[i].edge_node[10]].center,
                ]
            )
            box6 = Box(
                [
                    planes_outside[boxes[i].plane_node[1]].center,
                    len(out_points) - 1,
                    edges_outside[boxes[i].edge_node[6]].center,
                    planes_outside[boxes[i].plane_node[4]].center,
                    edges_outside[boxes[i].edge_node[9]].center,
                    planes_outside[boxes[i].plane_node[5]].center,
                    boxes[i].node[6],
                    edges_outside[boxes[i].edge_node[11]].center,
                ]
            )
            box7 = Box(
                [
                    len(out_points) - 1,
                    planes_outside[boxes[i].plane_node[3]].center,
                    planes_outside[boxes[i].plane_node[4]].center,
                    edges_outside[boxes[i].edge_node[7]].center,
                    planes_outside[boxes[i].plane_node[5]].center,
                    edges_outside[boxes[i].edge_node[10]].center,
                    edges_outside[boxes[i].edge_node[11]].center,
                    boxes[i].node[7],
                ]
            )
            new_plane = [
                Plane_outside(box0.node[2], box0.node[3], box0.node[7], box0.node[6]),
                Plane_outside(box0.node[1], box0.node[3], box0.node[7], box0.node[5]),
                Plane_outside(box0.node[4], box0.node[6], box0.node[7], box0.node[5]),
                Plane_outside(box3.node[0], box3.node[1], box3.node[5], box3.node[4]),
                Plane_outside(box3.node[0], box3.node[2], box3.node[6], box3.node[4]),
                Plane_outside(box3.node[4], box3.node[6], box3.node[7], box3.node[5]),
                Plane_outside(box5.node[0], box5.node[2], box5.node[3], box5.node[1]),
                Plane_outside(box5.node[0], box5.node[2], box5.node[6], box5.node[4]),
                Plane_outside(box5.node[2], box5.node[3], box5.node[7], box5.node[6]),
                Plane_outside(box6.node[0], box6.node[2], box6.node[3], box6.node[1]),
                Plane_outside(box6.node[0], box6.node[1], box6.node[5], box6.node[4]),
                Plane_outside(box6.node[1], box6.node[3], box6.node[7], box6.node[5]),
            ]
            for plane in new_plane:
                for edge_index in out_points[plane.node[0]].edges:
                    if (
                        edges_outside[edge_index].node[0] == plane.node[0]
                        and edges_outside[edge_index].node[1] == plane.node[1]
                    ) or (
                        edges_outside[edge_index].node[0] == plane.node[1]
                        and edges_outside[edge_index].node[1] == plane.node[0]
                    ):
                        plane.edge[0] = edge_index
                for edge_index in out_points[plane.node[1]].edges:
                    if (
                        edges_outside[edge_index].node[0] == plane.node[1]
                        and edges_outside[edge_index].node[1] == plane.node[2]
                    ) or (
                        edges_outside[edge_index].node[0] == plane.node[2]
                        and edges_outside[edge_index].node[1] == plane.node[1]
                    ):
                        plane.edge[1] = edge_index
                for edge_index in out_points[plane.node[2]].edges:
                    if (
                        edges_outside[edge_index].node[0] == plane.node[2]
                        and edges_outside[edge_index].node[1] == plane.node[3]
                    ) or (
                        edges_outside[edge_index].node[0] == plane.node[3]
                        and edges_outside[edge_index].node[1] == plane.node[2]
                    ):
                        plane.edge[2] = edge_index
                for edge_index in out_points[plane.node[3]].edges:
                    if (
                        edges_outside[edge_index].node[0] == plane.node[3]
                        and edges_outside[edge_index].node[1] == plane.node[0]
                    ) or (
                        edges_outside[edge_index].node[0] == plane.node[0]
                        and edges_outside[edge_index].node[1] == plane.node[3]
                    ):
                        plane.edge[3] = edge_index
                for ii in range(4):
                    if plane.edge[ii] == -1:
                        sys.exit("plane.edge[ii] == -1")
                for ii in range(4):
                    out_points[plane.node[ii]].planes.append(len(planes_outside))
                planes_outside.append(plane)
            box = [box0, box1, box2, box3, box4, box5, box6, box7]
            boxes[i].children = [len(boxes) + ii for ii in range(8)]
            for ii in range(8):
                set_boxes(box[ii], out_points, edges_outside, planes_outside)
                boxes.append(box[ii])

    return out_points, boxes, edges_outside, planes_outside
