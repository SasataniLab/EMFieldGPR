"""
Shared geometry classes and utilities for both Modeling and Visualize modules.
Consolidates duplicate class definitions across the codebase.
"""

import sys

import numpy as np


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
        self.mode = 0
        self.default_point = np.zeros(3)
        self.weight = np.zeros(0)
        self.parent = []
        self.edges = []
        self.planes = []  #
        for i in range(3):
            self.xyz[i] = point[i]


class Edge:
    """
    Class for edges that connect points in the mesh.

    Attributes
    ----------
    node : list
        Indices of the two points connected by this edge.
    normal : ndarray(3)
        Normal vector of the edge.
    children : list


    """

    def __init__(self, points, point1_index, point2_index):
        """
        Initialize an edge between two points.


        """
        self.node = [point1_index, point2_index]
        self.normal = points[point1_index].normal + points[point2_index].normal
        self.children = [-1, -1]  #

        vector = points[point2_index].xyz - points[point1_index].xyz
        vector /= np.linalg.norm(vector, ord=2)
        if np.linalg.norm(self.normal, ord=2) < 1e-7:  #
            print(
                point1_index,
                point2_index,
                points[point1_index].normal,
                points[point2_index].xyz,
            )
            sys.exit(["make invalid edge."])  #
        self.calculate_ray(vector)

    def calculate_ray(self, vector):
        """
        Calculate ray intersection with the edge.


        """
        s = -(
            vector[0] * self.normal[0]
            + vector[1] * self.normal[1]
            + vector[2] * self.normal[2]
        ) / (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
        ray_dir = s * vector + self.normal
        ray_dir = ray_dir / np.linalg.norm(ray_dir, ord=2)
        self.normal = ray_dir


class Edge_outside:
    """
    Edge class for outside points.


    """

    def __init__(self, index1, index2):
        """
        Initialize an outside edge between two points.


        """
        self.node = [index1, index2]  #
        self.children = [-1, -1]  #
        self.center = -1  #


class Plane:
    """
    Class for planes that define the mesh structure.

    Attributes
    ----------
    node : list
        Indices of the points that define this plane.
    edge : list

    normal : ndarray(3)
        Normal vector of the plane.
    children : list


    """

    def __init__(self, points, edges, points_index):
        """
        Initialize a plane with four points.


        """
        self.node = [points_index[0], points_index[1], points_index[2], points_index[3]]
        self.edge = [-1, -1, -1, -1]  #
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
            if self.edge[i] == -1:  #
                print(points_index)
                sys.exit(["make invalid plane."])
        self.normal = (
            points[points_index[0]].normal
            + points[points_index[1]].normal
            + points[points_index[2]].normal
            + points[points_index[3]].normal
        )
        self.children = [
            -1,
            -1,
            -1,
            -1,
        ]  #
        if np.linalg.norm(self.normal, ord=2) < 1e-7:  #
            sys.exit(["make invalid plane."])
        self.normal = self.normal / np.linalg.norm(self.normal, ord=2)


class Plane_outside:
    """
    Plane class for outside points.


    """

    def __init__(self, index1, index2, index3, index4):
        """
        Initialize an outside plane with four points.


        """
        self.node = [
            index1,
            index2,
            index3,
            index4,
        ]  #
        self.edge = [-1, -1, -1, -1]  #
        self.children = [
            -1,
            -1,
            -1,
            -1,
        ]  #
        self.center = -1  #


class Box:
    """
    Box class for mesh generation.


    """

    def __init__(self, points_index):
        """
        Initialize a box with eight corner points.


        """
        self.node = [
            points_index[0],
            points_index[1],
            points_index[2],
            points_index[3],
            points_index[4],
            points_index[5],
            points_index[6],
            points_index[7],
        ]  #
        self.children = [-1 for _ in range(8)]  #
        self.edge_node = []  #
        self.plane_node = []  #
