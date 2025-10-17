"""Second-level grid generation for electromagnetic field visualization.

This module provides functionality to create refined grid structures
for electromagnetic field analysis and visualization.


"""

import sys
from pathlib import Path

import numpy as np

from EMFieldML.Visualize.make_first_grid import (
    Box,
    Edge,
    Edge_outside,
    Plane,
    Plane_outside,
    Point,
    Point_outside,
)


def set_boxes(box, out_points, edges_outside, planes_outside):
    """Set box edges and planes based on predefined patterns.


    the topology of a box. Please verify if these indices are correct for the current
    box structure and update if the topology has changed.
    """
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
        now_list = [len(out_points[box.node[edge_l[i]]].edges) for i in range(2)]  #
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
        now_list = [len(out_points[box.node[plane_l[i]]].planes) for i in range(4)]  #
        focus = now_list.index(min(now_list))
        for plane_index in out_points[box.node[plane_l[focus]]].planes:
            flag = 0
            for jj in range(4):  #
                for iii in range(4):  #
                    if planes_outside[plane_index].node[jj] == box.node[plane_l[iii]]:
                        flag += 1
                        break
            if flag == 4:  #
                box.plane_node.append(plane_index)
                break
        if flag != 4:
            sys.exit("plane of box is none.")


def make_second_grid(
    input_path: Path,
    points: list[Point],
    edges: list[Edge],
    planes: list[Plane],
) -> tuple:
    """Create second-level grid for electromagnetic field visualization."""
    with Path(input_path).open() as f:
        data = f.readlines()
        coordinate_data = [float(row.rstrip("\n")) for row in data]  #

    for i in range(len(edges)):
        if edges[i].children[0] == -1:
            newpoint = Point(
                np.array(
                    [
                        coordinate_data[3 * len(points) + 0],
                        coordinate_data[3 * len(points) + 1],
                        coordinate_data[3 * len(points) + 2],
                    ]
                )
            )
            points.append(newpoint)
            newedge1 = Edge(edges[i].node[0], len(points) - 1)
            newedge2 = Edge(edges[i].node[1], len(points) - 1)
            edges.append(newedge1)
            edges[i].children[0] = len(edges) - 1
            edges.append(newedge2)
            edges[i].children[1] = len(edges) - 1

    for i in range(len(planes)):
        if planes[i].children[0] == -1:
            newpoint = Point(
                np.array(
                    [
                        coordinate_data[3 * len(points) + 0],
                        coordinate_data[3 * len(points) + 1],
                        coordinate_data[3 * len(points) + 2],
                    ]
                )
            )
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

            edges.append(Edge(plane_edge_point[0], len(points) - 1))
            edges.append(Edge(plane_edge_point[1], len(points) - 1))
            edges.append(Edge(plane_edge_point[2], len(points) - 1))
            edges.append(Edge(plane_edge_point[3], len(points) - 1))

            planes[i].children = [
                len(planes),
                len(planes) + 1,
                len(planes) + 2,
                len(planes) + 3,
            ]
            planes.append(
                Plane(
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
    """Create second-level out points for grid generation."""
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
                        for idx in range(4):  #
                            if planes[j].node[k] == new_point_index[idx]:
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
                        for idx in range(4):  #
                            if planes[j].node[k] == new_point_index[idx]:
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
